# FAQ

**Is this faster than TP2?**

It uses the third card. On the reference host the production profile reaches
124.3 tok/s single-stream and 635.3 tok/s at 16 streams, with a 1,011,875-token
KV pool. TP2 can be a better choice if you only have two cards; it is a
different recipe.

**Why not pad MoE 640 to 768?**

It costs about 4.5 GiB per card and changes the expert shape. EP3 keeps whole
experts per GPU (171/171/170), so no padding is needed. See
[design-notes.md](design-notes.md).

**Why 48GB cards? Can I use 24GB 4090s?**

No. The model needs about 28 GiB/card before KV. Three 24GB cards cannot hold
it. The repository name says 3x4090, but the requirement is specifically the
48GB variant.

**Is vision supported?**

Not validated. The production path is text-only. The model is multimodal, but
this recipe does not claim image/video support.

**Why MTP=2 and not 3?**

MTP=2 is the best balance on this host. MTP=3 has a smaller KV pool and lower
16-way throughput (541.1 vs 644.5 tok/s in the BF16 sweep). MTP=0 is the
capacity profile.

**Why is FP8 MTP not much faster?**

The gain is KV capacity, not raw tok/s. FP8 MTP saves about 0.78 GiB/card and
adds 52,428 KV tokens; throughput is within noise of BF16 MTP.

**Why not use KV CPU offloading for more context?**

vLLM's offloading connector does not understand this model's QSA/HiSparse KV
specs. It fails at the spec assertion, and a partial workaround would risk
silent wrong-cache behavior. See [design-notes.md](design-notes.md).

**Why pin vLLM 0.30.0?**

The patches are exact-anchor edits. A newer vLLM will fail `--check`. Porting
is possible but must be validated, especially the EP and prefix-cache fixes.

**Can I expose this to the internet?**

Not directly. The default bind is loopback. Put it behind a TLS reverse proxy
with authentication. This repository does not include a tunnel or FRP setup.

**Does it work without systemd?**

Yes. `./start.sh` and `./stop.sh` are the tested path. The systemd unit is a
template.

**Can I use Docker?**

The scripts are bare-metal. A CUDA 12.8 + vLLM 0.30.0 image is a reasonable
packaging layer, but it is not shipped here. The driver/wheel note in
[compatibility.md](compatibility.md) applies either way.

**Why does the first long request take ~47s?**

That is prefill for a ~288K prompt. The same request with a warm prefix cache
takes ~2s. Prefill is compute-bound; decode is bandwidth-bound.

**Where are the model weights?**

Not here. The repository contains scripts and documentation only. Obtain the
checkpoint from its publisher and comply with its license.

**The README says AI-generated. Is that a problem?**

It is a warning, not a disclaimer of responsibility. The scripts were tested
on the reference host, but environments differ. Run `./scripts/smoke.sh` and
open an issue with `./scripts/doctor.sh` output if something fails.