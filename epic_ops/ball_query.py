from typing import Tuple, Optional
from contextlib import redirect_stdout
import io
from pathlib import Path

import open3d  # noqa: F401
import torch

with redirect_stdout(io.StringIO()):
    pass


def _ensure_open3d_torch_ops_loaded() -> None:
    if hasattr(torch.ops.open3d, "build_spatial_hash_table") and hasattr(torch.ops.open3d, "fixed_radius_search"):
        return

    package_root = Path(open3d.__file__).resolve().parent
    candidates = []
    if torch.cuda.is_available():
        candidates.append(package_root / "cuda" / "open3d_torch_ops.so")
    candidates.append(package_root / "cpu" / "open3d_torch_ops.so")

    last_error: Exception | None = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            torch.ops.load_library(str(path))
            torch.classes.load_library(str(path))
            if hasattr(torch.ops.open3d, "build_spatial_hash_table") and hasattr(torch.ops.open3d, "fixed_radius_search"):
                return
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if not (hasattr(torch.ops.open3d, "build_spatial_hash_table") and hasattr(torch.ops.open3d, "fixed_radius_search")):
        if last_error is not None:
            raise RuntimeError(f"Failed to load Open3D torch ops: {last_error}") from last_error
        raise RuntimeError("Open3D torch ops library was not found or did not register search ops")


@torch.no_grad()
def ball_query(
    points: torch.Tensor,
    query: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_offsets: torch.Tensor,
    radius: float,
    num_samples: int,
    point_labels: Optional[torch.Tensor] = None,
    query_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _ensure_open3d_torch_ops_loaded()
    points = points.contiguous()
    query = query.contiguous()
    batch_indices = batch_indices.contiguous()
    batch_offsets = batch_offsets.contiguous()

    if point_labels is not None:
        point_labels = point_labels.contiguous()

    if query_labels is not None:
        query_labels = query_labels.contiguous()

    return torch.ops.epic_ops.ball_query(
        points, query, batch_indices, batch_offsets, radius, num_samples,
        point_labels, query_labels,
    )


@torch.no_grad()
def ball_query_fast(
    points: torch.Tensor,
    points_batch_offsets: torch.Tensor,
    queries: torch.Tensor,
    queries_batch_offsets: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _ensure_open3d_torch_ops_loaded()
    (
        hash_table_index, hash_table_cell_splits, hash_table_splits
    ) = torch.ops.open3d.build_spatial_hash_table(
        points=points,
        radius=radius,
        points_row_splits=points_batch_offsets,
        hash_table_size_factor=1 / 32,
        max_hash_table_size=33554432
    )

    (
        neighbors_index, neighbors_row_splits, _
    ) = torch.ops.open3d.fixed_radius_search(
        points=points,
        queries=queries,
        radius=radius,
        points_row_splits=points_batch_offsets,
        queries_row_splits=queries_batch_offsets,
        hash_table_splits=hash_table_splits,
        hash_table_index=hash_table_index,
        hash_table_cell_splits=hash_table_cell_splits,
        index_dtype=3,
        metric="L2",
        ignore_query_point=False,
        return_distances=False
    )

    return neighbors_index, neighbors_row_splits
