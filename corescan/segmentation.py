import numpy as np
from scipy import ndimage
from .core import InputError, number, integer, vector


def threshold(data, valid, lower, upper, invert=False):
    lower = number(lower, '阈值下限')
    upper = number(upper, '阈值上限', lower)
    foreground = (data >= lower) & (data <= upper)
    if invert:
        foreground = ~foreground
    foreground &= valid
    return {'volume': foreground.astype(np.float32), 'valid': valid.copy(), 'meta_update': {'unit': 'binary'}, 'report': {'lower': lower, 'upper': upper, 'invert': bool(invert), 'foreground_voxels': int(foreground.sum()), 'valid_voxels': int(valid.sum())}}


def physical_ball(radius_mm, spacing):
    radius = number(radius_mm, '半径mm', .00001, 50)
    extents = np.ceil(radius / np.asarray(spacing)).astype(int)
    if np.prod(2 * extents + 1) > 250000:
        raise InputError('结构元素过大，请减少半径')
    grids = np.meshgrid(*[np.arange(-n, n + 1) * step for n, step in zip(extents, spacing)], indexing='ij')
    return sum(grid * grid for grid in grids) <= radius * radius + 1e-12


def morphology(data, valid, meta, operation='opening', radius_mm=.4):
    foreground = (data > .5) & valid
    ball = physical_ball(radius_mm, meta['spacing'])
    if operation == 'opening':
        out = ndimage.binary_opening(foreground, structure=ball)
    elif operation == 'closing':
        out = ndimage.binary_closing(foreground, structure=ball)
    elif operation == 'dilation':
        out = ndimage.binary_dilation(foreground, structure=ball)
    elif operation == 'erosion':
        out = ndimage.binary_erosion(foreground, structure=ball)
    elif operation == 'fill_holes':
        out = ndimage.binary_fill_holes(foreground)
    else:
        raise InputError('形态操作不支持')
    out &= valid
    return {'volume': out.astype(np.float32), 'valid': valid.copy(), 'meta_update': {'unit': 'binary'}, 'report': {'operation': operation, 'radius_mm': radius_mm, 'before_voxels': int(foreground.sum()), 'after_voxels': int(out.sum()), 'note': '采用零边界；无效区始终不作为前景，孔洞填充仅限有界三维空腔。'}}


def face_area(foreground, spacing):
    total = 0.
    for axis in range(3):
        pad = [(0, 0)] * 3
        pad[axis] = (1, 1)
        transitions = np.diff(np.pad(foreground.astype(np.int8), pad), axis=axis)
        total += np.count_nonzero(transitions) * np.prod([spacing[i] for i in range(3) if i != axis])
    return float(total)


def components(data, valid, meta, connectivity=6, minimum_volume=0):
    connectivity = integer(connectivity, '连通性', 6, 26)
    if connectivity not in (6, 18, 26):
        raise InputError('连通性须为6、18或26')
    minimum_volume = number(minimum_volume, '最小体积mm³', 0)
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    foreground = (data > .5) & valid
    labels, count = ndimage.label(foreground, structure=structure)
    if count > 50000:
        raise InputError('连通对象超过50000，请先清理噪声或调整分区')
    sizes = np.bincount(labels.ravel())
    voxel_volume = float(np.prod(meta['spacing']))
    candidates = [i for i in range(1, count + 1) if sizes[i] * voxel_volume >= minimum_volume]
    candidates.sort(key=lambda label: (-sizes[label], label))
    mapping = np.zeros(count + 1, np.int32)
    rows = []
    slices = ndimage.find_objects(labels)
    spacing = np.asarray(meta['spacing'])
    origin = np.asarray(meta['origin'])
    invalid_neighbor = ndimage.binary_dilation(~valid, structure=structure)
    for new_id, old_id in enumerate(candidates, 1):
        mapping[old_id] = new_id
        bounding = slices[old_id - 1]
        local = labels[bounding] == old_id
        offset = np.array([s.start for s in bounding])
        coords = np.argwhere(local) + offset
        physical = coords * spacing + origin
        center = physical.mean(axis=0)
        covariance = np.cov(physical.T, bias=True) if len(coords) > 1 else np.zeros((3, 3))
        eig = np.maximum(np.linalg.eigvalsh(covariance), 0)
        touch_boundary = any(np.any(coords[:, axis] == 0) or np.any(coords[:, axis] == data.shape[axis] - 1) for axis in range(3))
        touch_invalid = bool(np.any(invalid_neighbor[tuple(coords.T)]))
        rows.append({'label': new_id, 'voxels': int(sizes[old_id]), 'volume_mm3': float(sizes[old_id] * voxel_volume), 'center_z_mm': float(center[0]), 'center_y_mm': float(center[1]), 'center_x_mm': float(center[2]), 'bbox_zyx': [[int(coords[:, a].min()), int(coords[:, a].max() + 1)] for a in range(3)], 'principal_rms_mm': np.sqrt(eig[::-1]).tolist(), 'voxel_face_area_mm2': face_area(local, spacing), 'touches_volume_boundary': touch_boundary, 'touches_invalid_support': touch_invalid, 'complete': not touch_boundary and not touch_invalid})
    out = mapping[labels]
    return {'volume': out.astype(np.float32), 'valid': valid.copy(), 'meta_update': {'unit': 'labels'}, 'tables': {'components': rows}, 'report': {'input_components': count, 'retained_components': len(rows), 'connectivity': connectivity, 'minimum_volume_mm3': minimum_volume, 'voxel_volume_mm3': voxel_volume, 'foreground_volume_mm3': float(np.count_nonzero(out) * voxel_volume), 'area_note': '体素面面积，非亚体素平滑曲面面积；边界/无效邻域接触另标为不完整。'}}


def select_labels(data, valid, labels):
    if not isinstance(labels, list) or not labels:
        raise InputError('至少选择一个正标签')
    selected = [integer(value, '标签', 1, 50000) for value in labels]
    present = set(np.unique(data[valid]).astype(int).tolist())
    if any(label not in present for label in selected):
        raise InputError('包含不存在的标签')
    foreground = np.isin(data, selected) & valid
    return {'volume': foreground.astype(np.float32), 'valid': valid.copy(), 'meta_update': {'unit': 'binary'}, 'report': {'labels': selected, 'foreground_voxels': int(foreground.sum())}}


def exclude_box(data, valid, bounds):
    if not isinstance(bounds, list) or len(bounds) != 6:
        raise InputError('排除区需要[z0,z1,y0,y1,x0,x1]')
    ranges = []
    for axis in range(3):
        lo = integer(bounds[axis * 2], '排除起点', 0, data.shape[axis] - 1)
        hi = integer(bounds[axis * 2 + 1], '排除终点', lo + 1, data.shape[axis])
        ranges.append(slice(lo, hi))
    good = valid.copy()
    good[tuple(ranges)] = False
    return {'volume': data.copy(), 'valid': good, 'report': {'excluded_bounds': bounds, 'newly_excluded_voxels': int(valid.sum() - good.sum()), 'valid_voxels': int(good.sum()), 'raw_values_preserved': True}}


def region_stats(data, valid, meta, bounds):
    from .volume_ops import crop, histogram
    result = crop(data, valid, meta, bounds)
    stats = histogram(result['volume'], result['valid'])
    stats['report'].update({'bounds': bounds, 'valid_volume_mm3': int(result['valid'].sum()) * float(np.prod(meta['spacing'])), 'origin_mm': result['meta']['origin']})
    return stats
