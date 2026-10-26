#!/usr/bin/env python3
"""02g: QSA side-cache 支持 fused multi-step draft decode (幂等) —— 上游 PR #58449 的 0.30.0 移植

背景: docs/01 阶段 7/8 判死单机 MTP 的直接原因是 0.30 的 QSA 注意力后端
      (QWEN4_EXP_EXP_QSA_STATE) 没声明 supports_draft_decode_metadata_update,
      speculator 每 draft step 重建 attention metadata (启动日志
      "Fused multi-step draft decode is not supported ... falling back"),
      每次 draft 额外 ~77ms 固定开销。
改法: 上游 PR #58449 (open, 未合入任何 release) 给 QSAMetadataBuilder 补上
      该能力声明 + update_draft_decode_metadata 原地重跑 metadata kernel。
      本脚本把该 PR 的 qsa_cache.py 改动 port 到 0.30.0 (结构完全一致, 干净套用)。
效果: fused draft metadata 生效 (fallback 日志从 1 条 -> 0 条), 解锁 MTP
      高并发备选 (4 流聚合 +77%)。但单机单流仍负收益 (draft 单次前向
      比主模型一步贵 2-3x, 见 docs/01), MTP 生产默认仍关闭。
备份: 首次运行时 qsa_cache.py.bak-speccache
验证: --selftest 跑 update_draft_decode_metadata == build 的逐位对照
      (plain/compressed/circular 三布局)
"""
import argparse
import ast
import os
import shutil
import sys

def _default_sp():
    import glob, os
    hits = glob.glob(os.path.expanduser(
        "~/vllm-venv/lib/python3.*/site-packages/vllm/models/qwen4_exp/common/qsa_cache.py"))
    if not hits:
        raise SystemExit("!! 找不到 ~/vllm-venv 里的 qsa_cache.py，请显式传 qsa_cache.py 路径")
    return hits[0]
MARKER = "supports_draft_decode_metadata_update = HAS_TRITON"

# ---- 原始 0.30.0 build_qsa_metadata_triton (未打, 从 PR diff 的 '-' 行还原) ----
ORIG_TRITON = '''def build_qsa_metadata_triton(
    common_attn_metadata: CommonAttentionMetadata,
    token_to_req_buffer: torch.Tensor,
    logical_positions_buffer: torch.Tensor,
    visible_blocks_buffer: torch.Tensor,
    slot_mapping_buffer: torch.Tensor,
    *,
    storage_block_size: int,
    compress_ratio: int,
    circular_buffer_size: int = 0,
    k_work_metadata_buffer: torch.Tensor | None = None,
    request_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build QSA side-cache and optional pre-indexer work metadata."""
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
    token_to_req = token_to_req_buffer[:num_tokens]
    logical_positions = logical_positions_buffer[:num_tokens]
    visible_blocks = visible_blocks_buffer[:num_tokens]
    slot_mapping = slot_mapping_buffer[:num_tokens]
    num_reqs = common_attn_metadata.query_start_loc.shape[0] - 1
    assert num_reqs > 0

    if k_work_metadata_buffer is not None:
        if request_capacity is None:
            request_capacity = num_reqs
        assert request_capacity >= num_reqs
        # Pad for tl.arange while keeping the scan width stable across live batches.
        request_scan_size = 1 << int(math.ceil(math.log2(request_capacity)))
        max_num_work = k_work_metadata_buffer.shape[0]
    else:
        request_scan_size = 1
        max_num_work = 0

    if num_tokens == 0 and k_work_metadata_buffer is None:
        return token_to_req, logical_positions, visible_blocks, slot_mapping

    block_table = common_attn_metadata.block_table_tensor
    num_search_steps = int(math.ceil(math.log2(num_reqs)))
    work_search_steps = int(math.ceil(math.log2(num_reqs)))
    # The same grid covers token tiles and, for the compressed cache, work tiles.
    num_token_blocks = cdiv(num_tokens, 128)
    num_work_blocks = (
        cdiv(max_num_work, 256) if k_work_metadata_buffer is not None else 0
    )
    _build_qsa_metadata_kernel[(max(num_token_blocks, num_work_blocks, 1),)](
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        common_attn_metadata.slot_mapping,
        block_table,
        token_to_req,
        logical_positions,
        visible_blocks,
        slot_mapping,
        k_work_metadata_buffer,
        block_table.stride(0),
        block_table.stride(1),
        num_reqs,
        num_mapped_tokens,
        num_tokens,
        max_num_work,
        num_search_steps,
        work_search_steps,
        storage_block_size,
        compress_ratio,
        circular_buffer_size,
        block_table.shape[1],
        launch_pdl=_metadata_launch_pdl(),
        TOKEN_BLOCK_SIZE=128,
        REQUEST_SCAN_SIZE=request_scan_size,
        WORK_BLOCK_SIZE=256,
        num_warps=4,
    )
    if circular_buffer_size == 0 and compress_ratio == 1:
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
    return token_to_req, logical_positions, visible_blocks, slot_mapping'''

# ---- 打后: 抽出 _launch_qsa_metadata_kernel, build_*_triton 变成薄封装 ----
PATCHED = '''def _launch_qsa_metadata_kernel(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    common_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_work_metadata_buffer: torch.Tensor | None,
    *,
    num_mapped_tokens: int,
    storage_block_size: int,
    compress_ratio: int,
    circular_buffer_size: int,
    request_capacity: int | None,
) -> None:
    """Fill QSA metadata in place; capture-safe given fixed shapes and scalars."""
    num_tokens = token_to_req.shape[0]
    num_reqs = query_start_loc.shape[0] - 1
    assert num_reqs > 0

    if k_work_metadata_buffer is not None:
        if request_capacity is None:
            request_capacity = num_reqs
        assert request_capacity >= num_reqs
        # Pad for tl.arange while keeping the scan width stable across live batches.
        request_scan_size = 1 << int(math.ceil(math.log2(request_capacity)))
        max_num_work = k_work_metadata_buffer.shape[0]
    else:
        request_scan_size = 1
        max_num_work = 0

    num_search_steps = int(math.ceil(math.log2(num_reqs)))
    work_search_steps = int(math.ceil(math.log2(num_reqs)))
    # The same grid covers token tiles and, for the compressed cache, work tiles.
    num_token_blocks = cdiv(num_tokens, 128)
    num_work_blocks = (
        cdiv(max_num_work, 256) if k_work_metadata_buffer is not None else 0
    )
    _build_qsa_metadata_kernel[(max(num_token_blocks, num_work_blocks, 1),)](
        query_start_loc,
        seq_lens,
        common_slot_mapping,
        block_table,
        token_to_req,
        logical_positions,
        visible_blocks,
        slot_mapping,
        k_work_metadata_buffer,
        block_table.stride(0),
        block_table.stride(1),
        num_reqs,
        num_mapped_tokens,
        num_tokens,
        max_num_work,
        num_search_steps,
        work_search_steps,
        storage_block_size,
        compress_ratio,
        circular_buffer_size,
        block_table.shape[1],
        launch_pdl=_metadata_launch_pdl(),
        TOKEN_BLOCK_SIZE=128,
        REQUEST_SCAN_SIZE=request_scan_size,
        WORK_BLOCK_SIZE=256,
        num_warps=4,
    )


def build_qsa_metadata_triton(
    common_attn_metadata: CommonAttentionMetadata,
    token_to_req_buffer: torch.Tensor,
    logical_positions_buffer: torch.Tensor,
    visible_blocks_buffer: torch.Tensor,
    slot_mapping_buffer: torch.Tensor,
    *,
    storage_block_size: int,
    compress_ratio: int,
    circular_buffer_size: int = 0,
    k_work_metadata_buffer: torch.Tensor | None = None,
    request_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build QSA side-cache and optional pre-indexer work metadata."""
    num_tokens = common_attn_metadata.num_actual_tokens
    token_to_req = token_to_req_buffer[:num_tokens]
    logical_positions = logical_positions_buffer[:num_tokens]
    visible_blocks = visible_blocks_buffer[:num_tokens]
    slot_mapping = slot_mapping_buffer[:num_tokens]

    if num_tokens == 0 and k_work_metadata_buffer is None:
        return token_to_req, logical_positions, visible_blocks, slot_mapping

    _launch_qsa_metadata_kernel(
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        common_attn_metadata.slot_mapping,
        common_attn_metadata.block_table_tensor,
        token_to_req,
        logical_positions,
        visible_blocks,
        slot_mapping,
        k_work_metadata_buffer,
        num_mapped_tokens=int(common_attn_metadata.query_start_loc_cpu[-1]),
        storage_block_size=storage_block_size,
        compress_ratio=compress_ratio,
        circular_buffer_size=circular_buffer_size,
        request_capacity=request_capacity,
    )
    if circular_buffer_size == 0 and compress_ratio == 1:
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
    return token_to_req, logical_positions, visible_blocks, slot_mapping'''

# metadata 字段 + builder 能力声明 + build 传参 + update 方法
ORIG_META_FIELDS = '''    visible_blocks: torch.Tensor
    k_work_metadata: torch.Tensor
    num_actual_tokens: int'''
PATCHED_META_FIELDS = '''    visible_blocks: torch.Tensor
    k_work_metadata: torch.Tensor
    common_slot_mapping: torch.Tensor
    num_mapped_tokens: int
    num_actual_tokens: int'''

ORIG_BUILDER_DECL = '''    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__('''
PATCHED_BUILDER_DECL = '''    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # pico-llm 02g: port of upstream PR #58449 —— 让 QSA builder 支持 fused
    # multi-step draft decode (MTP speculator 跨 draft step 复用一份 metadata,
    # 不再每步重建)。单机单流仍负收益, 仅解锁 MTP 高并发备选。
    supports_draft_decode_metadata_update = HAS_TRITON

    def __init__('''

ORIG_BUILD_RET = '''            visible_blocks=visible_blocks,
            k_work_metadata=k_work_metadata,
            num_actual_tokens=num_tokens,'''
PATCHED_BUILD_RET = '''            visible_blocks=visible_blocks,
            k_work_metadata=k_work_metadata,
            common_slot_mapping=common_attn_metadata.slot_mapping,
            num_mapped_tokens=int(common_attn_metadata.query_start_loc_cpu[-1]),
            num_actual_tokens=num_tokens,'''

ORIG_UPDATE_ANCHOR = '''            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )


class QSAStateBackend(AttentionBackend):'''
PATCHED_UPDATE = '''            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )

    def update_draft_decode_metadata(self, metadata: QSAForwardMetadata) -> None:
        if metadata.num_actual_tokens == 0:
            return
        build_k_work = not self.is_circular_buffer and self.compress_ratio != 1
        _launch_qsa_metadata_kernel(
            metadata.query_start_loc,
            metadata.seq_lens,
            metadata.common_slot_mapping,
            metadata.block_table,
            metadata.token_to_req,
            metadata.logical_positions,
            metadata.visible_blocks,
            metadata.slot_mapping,
            metadata.k_work_metadata if build_k_work else None,
            num_mapped_tokens=metadata.num_mapped_tokens,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            circular_buffer_size=(
                self.kv_cache_spec.block_size if self.is_circular_buffer else 0
            ),
            request_capacity=self.request_capacity if build_k_work else None,
        )


class QSAStateBackend(AttentionBackend):'''

REPLACEMENTS = [
    (ORIG_TRITON, PATCHED),
    (ORIG_META_FIELDS, PATCHED_META_FIELDS),
    (ORIG_BUILDER_DECL, PATCHED_BUILDER_DECL),
    (ORIG_BUILD_RET, PATCHED_BUILD_RET),
    (ORIG_UPDATE_ANCHOR, PATCHED_UPDATE),
]


def already_patched(src: str) -> bool:
    return MARKER in src


def apply(src: str) -> str:
    for old, new in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"!! 替换锚点出现 {n} 次 (期望 1), 文件结构与预期不符, 拒绝写入:\n"
                f"   锚点开头: {old[:60]!r}"
            )
        src = src.replace(old, new, 1)
    return src


def selftest() -> int:
    """update_draft_decode_metadata 与 build() 的逐位对照 (plain/compressed/circular)。"""
    import torch
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder

    device = torch.device("cuda")

    def make_builder(b, cr, circular, block_size=4, storage=4):
        b.compress_ratio = cr
        b.reorder_batch_threshold = 1
        b.is_circular_buffer = circular
        b.kv_cache_spec = SimpleNamespace(block_size=block_size)
        b.storage_block_size = storage
        b.token_to_req_buffer = torch.empty(8, dtype=torch.int32, device=device)
        b.slot_mapping_buffer = torch.empty(8, dtype=torch.int64, device=device)
        b.logical_positions_buffer = torch.empty(8, dtype=torch.int64, device=device)
        b.visible_blocks_buffer = torch.empty(8, dtype=torch.int32, device=device)
        b.request_capacity = 8
        b.k_work_metadata_buffer = torch.empty(
            0 if circular else 8, 2, dtype=torch.int32, device=device
        )

    qsl = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    def make_common(seq_lens):
        return SimpleNamespace(
            num_actual_tokens=3, num_reqs=3, max_query_len=1,
            max_seq_len=int(seq_lens.max()), query_start_loc=qsl,
            query_start_loc_cpu=qsl.cpu(), seq_lens=seq_lens,
            slot_mapping=torch.tensor([5, 9, 20], dtype=torch.int64, device=device),
            block_table_tensor=torch.tensor(
                [[3, 1], [0, 4], [2, 5]], dtype=torch.int32, device=device
            ),
            token_to_req_indices=lambda buf: buf.copy_(token_to_req),
        )

    ok_all = True
    for layout, cr, circular, storage in [
        ("plain", 1, False, 4),
        ("compressed", 4, False, 16),
        ("circular", 1, True, 4),
    ]:
        seq = torch.tensor([7, 11, 16], dtype=torch.int32, device=device)
        b = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
        make_builder(b, cr, circular, storage=storage)
        meta = b.build(0, make_common(seq))
        before = meta.logical_positions.clone()
        seq.add_(1)
        b.update_draft_decode_metadata(meta)
        eb = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
        make_builder(eb, cr, circular, storage=storage)
        exp = eb.build(0, make_common(seq.clone()))
        ok = (
            not torch.equal(meta.logical_positions, before)
            and all(
                torch.equal(getattr(meta, f), getattr(exp, f))
                for f in ("token_to_req", "logical_positions", "visible_blocks",
                          "slot_mapping", "k_work_metadata")
            )
        )
        ok_all &= ok
        print(f"  [{layout}] update==rebuild 且状态已变: {ok}")
    print("==> selftest " + ("通过" if ok_all else "失败"))
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="跑 update==rebuild 对照后退出")
    ap.add_argument("target", nargs="?", default=None,
                    help="qsa_cache.py 路径（默认自动探测 ~/vllm-venv）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    target = args.target or _default_sp()
    bak = target + ".bak-speccache"
    src = open(target).read()
    if already_patched(src):
        print("==> 补丁已存在 (supports_draft_decode_metadata_update), 跳过")
        return 0
    if not os.path.exists(bak):
        shutil.copy2(target, bak)
        print(f"==> 备份: {os.path.basename(bak)}")
    new = apply(src)
    ast.parse(new)
    open(target, "w").write(new)
    print("==> 02g 已写入, 语法检查通过; 验证: ./02g-patch-vllm-qsa-spec-cache.py --selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
