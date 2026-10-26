# Design notes

This is the short version of the deployment log. It records what was tried,
what failed, and why the current configuration looks the way it does.

## 1. Why not pad MoE 640 to 768?

TP3 needs the routed-expert intermediate dimension to split cleanly. 640 / 3
is 213.3, not 128-aligned. Padding to 768 gives 256 per rank and makes the
kernels happy, but:

- it adds ~4.5 GiB per card;
- it changes the expert shape, so the checkpoint must be converted;
- it spends VRAM that could be KV;
- it introduces a quality question that is hard to test exhaustively.

EP3 was chosen instead. Each GPU owns whole experts (171/171/170), so the
intermediate dimension is never split. No padding, no extra 4.5 GiB.

## 2. EP3 initially produced wrong output

The first EP3 runs were coherent for short prompts but failed arithmetic and
occasionally repeated tokens. The cause was in vLLM's MARLIN MoE kernel: it
multiplied the top-k router weights inside the kernel even when `expert_map`
was set, and the EP path also multiplied them outside. The weights were
applied twice.

`02k` changes the kernel call to `mul_topk_weights=False` when `expert_map` is
present and applies the multiplication after the kernel. Against an FP32
reference, relative error went from 0.47–1.16 to ~0.0016. A 20-question
three-number multiplication set passed 20/20 after the fix.

This is why the smoke test is arithmetic-heavy. A model that chats fluently
can still be numerically wrong under EP.

## 3. TP3 model patch

`02i` is the largest patch. It touches six files:

- `qwen_gdn_linear_attn.py`: local TP1 for GDN, full conv/qkvz/ba/out and
  `A_log`/`dt_bias`.
- `qwen4_exp/nvidia/model.py`: embed/lm_head padding 192, PLE divisor 384.
- `qwen4_exp/nvidia/qsa.py`: full 24 Q heads + 2 KV heads on every rank.
- `qwen4_exp/nvidia/mtp.py`: MTP shape changes for TP3.
- `qwen3_next.py`: shared-expert replication when TP3 cannot align.
- `ngram_embedding.py`: PLE shard read/padding for divisor 384.

Without these, TP3 either refuses to load or produces shape errors.

## 4. FP8 side layers

The NVFP4 checkpoint leaves ~15 GiB of dense side layers in BF16. Those layers
are read on every token, so they dominate decode bandwidth. Converting the 156
GDN/QSA side tensors to blockwise FP8 (the default; shared experts are opt-in
via `--include-shared-experts`):

- drops model load from 28.74 to 27.96 GiB/card;
- adds 52,428 KV tokens;
- keeps worst per-tensor relative error around 3.54%.

The conversion is non-destructive: a new overlay directory is created, with
symlinks for unchanged shards. The runtime shim only activates when
`VLLM_FP8_HYBRID=1`.

## 5. FP8 MTP experts

The MTP layer has its own 512 experts. Quantizing them to 128x128 block FP8:

- writes a 2.34 GiB overlay;
- removes the old packed BF16 MTP shard from the index;
- drops model load by another 0.78 GiB/card;
- adds 52,428 KV tokens over BF16 MTP;
- keeps cosine similarity at 0.9996 for gate/up/down on expert 0.

Throughput is effectively unchanged. The reason to do it is KV, not tok/s.

## 6. MTP was "dead" twice

Early tests concluded MTP was a loss. Both conclusions were wrong:

1. The benchmark counted SSE chunks. With speculative decoding, one chunk can
   contain multiple tokens, so the "tok/s" was really steps/s.
2. vLLM 0.30's QSA backend did not support fused draft metadata; it rebuilt
   attention metadata for every draft step, adding ~77 ms per step.

`02g` ports the fused-draft support from upstream PR #58449. With the patch,
the fallback log disappears and MTP=2 becomes positive at high concurrency.

## 7. Prefix cache was silently wrong

The mamba hybrid scheduler uses the smallest KV-group block size for chunk
alignment, but the state slot seed needs the mamba block size. On a cache hit,
the state was read from the wrong column and restored as zeros. The output
looked plausible but changed on repeated prompts.

`02h` fixes both the worker seed and the scheduler split. The repeat-prefix
test is mandatory after any vLLM upgrade.

## 8. Rejected options

| Option | Why rejected |
|---|---|
| MoE768 padding | +4.5 GiB/card, quality risk, extra conversion |
| MTP=3 | Lower 16-way throughput than MTP=2 on this host |
| KV CPU offloading | vLLM offloading connector does not understand QSA/HiSparse KV specs; silent wrong-cache risk |
| 24GB 4090 | Model needs ~28 GiB/card before KV |
| Vision in production | Not validated; text-only is the supported path |
| Multi-node / Strata | Different architecture; not part of this single-host recipe |

## 9. What would change the conclusions

- A vLLM release with native Qwen4Exp TP3/EP3 support would remove most of
  `02i`/`02k`.
- An upstream MARLIN fix would remove `02k`.
- An upstream QSA fused-draft fix would remove `02g`.
- An upstream mamba block-size fix would remove `02h`.

Until then, pin vLLM 0.30.0 and run the `--check` paths.