#!/usr/bin/env python3
"""Enable Qwen4Exp TP3 with EP, replicated GDN/QSA KV, and padded vocab."""

import argparse
import ast
import shutil
from pathlib import Path

GDN_REPLACEMENTS = [
    (
        """from vllm.distributed import (
    divide,
)""",
        """from vllm.distributed import (
    divide,
    get_tensor_model_parallel_world_size,
)""",
    ),
    (
        """        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads""",
        """        self.replicate_tp = get_tensor_model_parallel_world_size() == 3
        if self.replicate_tp:
            self.tp_size = 1
            self.tp_rank = 0
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads""",
    ),
    (
        """            bias=False,
            prefix=f"{prefix}.conv1d",
        )""",
        """            bias=False,
            prefix=f"{prefix}.conv1d",
            disable_tp=self.replicate_tp,
        )""",
    ),
    (
        """            bias=False,
            input_is_parallel=True,
            reduce_results=reduce_results,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )""",
        """            bias=False,
            input_is_parallel=not self.replicate_tp,
            reduce_results=reduce_results,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=self.replicate_tp,
        )""",
    ),
    (
        """        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})""",
        """        if not self.replicate_tp:
            set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
            set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})""",
    ),
    (
        """            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(""",
        """            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=self.replicate_tp,
        )

    def create_ba_proj(""",
    ),
    (
        "            disable_tp=self.maybe_disable_tp(quant_config),",
        "            disable_tp=self.replicate_tp or self.maybe_disable_tp(quant_config),",
    ),
]

MODEL_REPLACEMENTS = [
    (
        """        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)""",
        """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, config.hidden_size, padding_size=192
        )""",
    ),
    (
        """            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )""",
        """            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
            padding_size=192,
        )""",
    ),
    (
        """        tp_size = parallel_config.tensor_parallel_size
        num_spec = (""",
        """        tp_size = parallel_config.tensor_parallel_size
        if tp_size == 3:
            tp_size = 1
        num_spec = (""",
    ),
]

QSA_REPLACEMENTS = [
    (
        """        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size""",
        """        tp_size = get_tensor_model_parallel_world_size()
        self.replicate_tp = tp_size == 3
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = (
            self.total_num_heads if self.replicate_tp else self.total_num_heads // tp_size
        )""",
    ),
    (
        """        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)""",
        """        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        self.num_kv_heads = (
            self.total_num_kv_heads
            if self.replicate_tp or tp_size % self.total_num_kv_heads
            else max(1, self.total_num_kv_heads // tp_size)
        )""",
    ),
    (
        """            self.total_num_kv_heads,
            bias=False,
            quant_config=_fp8_hybrid_excluded(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )""",
        """            self.total_num_kv_heads,
            bias=False,
            quant_config=_fp8_hybrid_excluded(quant_config),
            prefix=f"{prefix}.qkv_proj",
            disable_tp=self.replicate_tp,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=not self.replicate_tp,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=self.replicate_tp,
        )""",
    ),
]
MTP_REPLACEMENTS = [
    (
        """        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)""",
        """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, self.hidden_size, padding_size=192
        )""",
    ),
    (
        """                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )""",
        """                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                    padding_size=192,
                )""",
    ),
    (
        """                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )""",
        """                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
                disable_tp=True,
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
                disable_tp=True,
            )""",
    ),
]
SHARED_EXPERT_REPLACEMENTS = [
    (
        """    if intermediate_size <= 0 or group_size is None:
        return False""",
        """    if tp_size == 3 and intermediate_size > 0 and intermediate_size % tp_size:
        return True
    if intermediate_size <= 0 or group_size is None:
        return False""",
    ),
]
PLE_REPLACEMENTS = [
    (
        """                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )""",
        """                checkpoint_vocab_size = (
                    (int(self.ngram_heads_vocab_sizes.sum()) + 127) // 128
                ) * 128
                checkpoint_shard_size = (
                    checkpoint_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * checkpoint_shard_size
                expected_rows = max(
                    0,
                    min(
                        checkpoint_shard_size,
                        checkpoint_vocab_size - checkpoint_start,
                    ),
                )""",
    ),
]


def patch_file(path: Path, replacements: list[tuple[str, str]], check: bool) -> None:
    original = path.read_text()
    patched = original
    for old, new in replacements:
        if new in patched:
            continue
        if patched.count(old) != 1:
            raise RuntimeError(f"expected one patch anchor in {path}: {old[:80]!r}")
        patched = patched.replace(old, new, 1)
    ast.parse(patched, filename=str(path))
    if patched == original:
        print(f"OK {path}")
        return
    if check:
        raise RuntimeError(f"not patched: {path}")
    backup = path.with_name(path.name + ".bak-tp3")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched)
    print(f"PATCHED {path} (backup: {backup})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.vllm_root
    if root is None:
        import vllm

        if vllm.__version__ != "0.30.0":
            raise SystemExit(f"unsupported vLLM version: {vllm.__version__}; expected 0.30.0")
        root = Path(vllm.__file__).parent

    patch_file(
        root / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        GDN_REPLACEMENTS,
        args.check,
    )
    patch_file(root / "models/qwen4_exp/nvidia/model.py", MODEL_REPLACEMENTS, args.check)
    patch_file(root / "models/qwen4_exp/nvidia/qsa.py", QSA_REPLACEMENTS, args.check)
    patch_file(root / "models/qwen4_exp/nvidia/mtp.py", MTP_REPLACEMENTS, args.check)
    patch_file(
        root / "model_executor/models/qwen3_next.py",
        SHARED_EXPERT_REPLACEMENTS,
        args.check,
    )
    patch_file(
        root / "models/qwen4_exp/nvidia/ngram_embedding.py",
        PLE_REPLACEMENTS,
        args.check,
    )


if __name__ == "__main__":
    main()
