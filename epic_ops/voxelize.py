from typing import List, Tuple
import os

import torch

from .expand import expand_csr


def _voxelize_single_batch(
    points: torch.Tensor,
    pt_features: torch.Tensor,
    voxel_size: torch.Tensor,
    points_range_min: torch.Tensor,
    points_range_max: torch.Tensor,
    reduction: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask = ((points >= points_range_min) & (points < points_range_max)).all(dim=1)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        empty_feat = pt_features.new_zeros((0, pt_features.shape[1]))
        empty_coords = torch.empty((0, points.shape[1]), dtype=torch.int32, device=points.device)
        empty_ids = torch.full((points.shape[0],), -1, dtype=torch.int64, device=points.device)
        return empty_feat, empty_coords, empty_ids.new_zeros((0,), dtype=torch.int64), empty_ids

    valid_points = points[valid_indices]
    valid_features = pt_features[valid_indices]

    voxel_coords_float = torch.floor((valid_points - points_range_min) / voxel_size)
    voxel_coords = voxel_coords_float.to(dtype=torch.int32)

    unique_coords, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
    num_voxels = unique_coords.shape[0]

    counts = torch.bincount(inverse_indices, minlength=num_voxels)
    if reduction == "mean":
        voxel_features = torch.zeros(
            (num_voxels, valid_features.shape[1]),
            dtype=valid_features.dtype,
            device=valid_features.device,
        )
        voxel_features.index_add_(0, inverse_indices, valid_features)
        voxel_features = voxel_features / counts.clamp_min(1).to(valid_features.dtype).unsqueeze(1)
    else:
        raise ValueError(f"Unsupported reduction in compatibility voxelize: {reduction}")

    sort_keys = inverse_indices.to(torch.int64) * valid_indices.shape[0] + torch.arange(
        valid_indices.shape[0], device=valid_indices.device, dtype=torch.int64
    )
    order = torch.argsort(sort_keys)
    voxel_point_indices = valid_indices[order]

    voxel_point_row_splits = torch.zeros((num_voxels + 1,), dtype=torch.int64, device=points.device)
    voxel_point_row_splits[1:] = counts.to(torch.int64).cumsum(0)

    voxel_batch_splits = torch.tensor([0, num_voxels], dtype=torch.int64, device=points.device)
    batch_indices, _ = expand_csr(voxel_batch_splits, unique_coords.shape[0])

    pc_voxel_id = torch.full((points.shape[0],), -1, dtype=torch.int64, device=points.device)
    pc_voxel_id[valid_indices] = inverse_indices.to(torch.int64)

    return voxel_features, unique_coords, batch_indices, pc_voxel_id


def _voxelize_multi_batch(
    points: torch.Tensor,
    pt_features: torch.Tensor,
    batch_offsets: torch.Tensor,
    voxel_size: torch.Tensor,
    points_range_min: torch.Tensor,
    points_range_max: torch.Tensor,
    reduction: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    all_voxel_features = []
    all_voxel_coords = []
    all_batch_indices = []
    pc_voxel_id = torch.full((points.shape[0],), -1, dtype=torch.int64, device=points.device)

    voxel_offset = 0
    for batch_id in range(batch_offsets.numel() - 1):
        begin = int(batch_offsets[batch_id].item())
        end = int(batch_offsets[batch_id + 1].item())
        batch_points = points[begin:end]
        batch_features = pt_features[begin:end]
        vf, vc, _vb, local_pc_voxel_id = _voxelize_single_batch(
            batch_points,
            batch_features,
            voxel_size,
            points_range_min,
            points_range_max,
            reduction,
        )
        all_voxel_features.append(vf)
        all_voxel_coords.append(vc)
        all_batch_indices.append(
            torch.full((vc.shape[0],), batch_id, dtype=torch.int64, device=points.device)
        )
        valid_local = local_pc_voxel_id >= 0
        pc_voxel_id_batch = torch.full_like(local_pc_voxel_id, -1)
        pc_voxel_id_batch[valid_local] = local_pc_voxel_id[valid_local] + voxel_offset
        pc_voxel_id[begin:end] = pc_voxel_id_batch
        voxel_offset += vc.shape[0]

    if not all_voxel_features:
        empty_feat = pt_features.new_zeros((0, pt_features.shape[1]))
        empty_coords = torch.empty((0, points.shape[1]), dtype=torch.int32, device=points.device)
        empty_batch = torch.empty((0,), dtype=torch.int64, device=points.device)
        return empty_feat, empty_coords, empty_batch, pc_voxel_id

    voxel_features = torch.cat(all_voxel_features, dim=0)
    voxel_coords = torch.cat(all_voxel_coords, dim=0)
    batch_indices = torch.cat(all_batch_indices, dim=0)
    return voxel_features, voxel_coords, batch_indices, pc_voxel_id


def voxelize_raw(
    points: torch.Tensor,
    pt_features: torch.Tensor,
    batch_offsets: torch.Tensor,
    voxel_size: List[float],
    points_range_min: List[float],
    points_range_max: List[float],
    reduction: str = "mean",
    max_points_per_voxel: int = 9223372036854775807,
    max_voxels: int = 9223372036854775807,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del max_points_per_voxel, max_voxels
    points = points.contiguous()
    pt_features = pt_features.contiguous()
    batch_offsets = batch_offsets.to(dtype=torch.int64, device=points.device).contiguous()
    voxel_size_t = torch.as_tensor(voxel_size, dtype=torch.float32, device=points.device).contiguous()
    points_range_min_t = torch.as_tensor(points_range_min, dtype=torch.float32, device=points.device).contiguous()
    points_range_max_t = torch.as_tensor(points_range_max, dtype=torch.float32, device=points.device).contiguous()

    if batch_offsets.numel() == 2:
        return _voxelize_single_batch(
            points,
            pt_features,
            voxel_size_t,
            points_range_min_t,
            points_range_max_t,
            reduction,
        )
    return _voxelize_multi_batch(
        points,
        pt_features,
        batch_offsets,
        voxel_size_t,
        points_range_min_t,
        points_range_max_t,
        reduction,
    )


def voxelize(
    points: torch.Tensor,
    pt_features: torch.Tensor,
    batch_offsets: torch.Tensor,
    voxel_size: torch.Tensor,
    points_range_min: torch.Tensor,
    points_range_max: torch.Tensor,
    reduction: str = "mean",
    max_points_per_voxel: int = 9223372036854775807,
    max_voxels: int = 9223372036854775807,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del max_points_per_voxel, max_voxels
    points = points.contiguous()
    pt_features = pt_features.contiguous()
    batch_offsets = batch_offsets.to(dtype=torch.int64, device=points.device).contiguous()
    voxel_size = voxel_size.to(dtype=torch.float32, device=points.device).contiguous()
    points_range_min = points_range_min.to(dtype=torch.float32, device=points.device).contiguous()
    points_range_max = points_range_max.to(dtype=torch.float32, device=points.device).contiguous()

    if bool(int(os.environ.get("EPIC_OPS_DEBUG", "0"))):
        print(
            "[epic_ops.voxelize]",
            "points", points.dtype, points.device, points.shape, points.is_contiguous(),
            "row_splits", batch_offsets.dtype, batch_offsets.device, batch_offsets.shape, batch_offsets.is_contiguous(),
            "voxel_size", voxel_size.dtype, voxel_size.device, voxel_size.shape, voxel_size.is_contiguous(),
            "min", points_range_min.dtype, points_range_min.device, points_range_min.shape, points_range_min.is_contiguous(),
            "max", points_range_max.dtype, points_range_max.device, points_range_max.shape, points_range_max.is_contiguous(),
        )

    if batch_offsets.numel() == 2:
        return _voxelize_single_batch(
            points,
            pt_features,
            voxel_size,
            points_range_min,
            points_range_max,
            reduction,
        )
    return _voxelize_multi_batch(
        points,
        pt_features,
        batch_offsets,
        voxel_size,
        points_range_min,
        points_range_max,
        reduction,
    )
