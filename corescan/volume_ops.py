import numpy as np
from scipy import ndimage
from .core import InputError, axis_index, number, integer, vector, summary_stats


def section(data, valid, axis='z', index=None):
    dim = axis_index(axis)
    index = integer(data.shape[dim] // 2 if index is None else index, '切片位置', 0, data.shape[dim] - 1)
    return np.take(data, index, axis=dim), np.take(valid, index, axis=dim)


def projection(data, valid, meta, axis='z', method='maximum', start=None, end=None):
    dim = axis_index(axis)
    start = integer(0 if start is None else start, '开始位置', 0, data.shape[dim] - 1)
    end = integer(data.shape[dim] if end is None else end, '结束位置', start + 1, data.shape[dim])
    sl = [slice(None)] * 3
    sl[dim] = slice(start, end)
    a, m = data[tuple(sl)], valid[tuple(sl)]
    count = m.sum(axis=dim)
    if method == 'maximum':
        result = np.max(np.where(m, a, -np.inf), axis=dim)
    elif method == 'minimum':
        result = np.min(np.where(m, a, np.inf), axis=dim)
    elif method in ('mean', 'integral'):
        result = np.sum(np.where(m, a, 0), axis=dim, dtype=np.float64)
        if method == 'mean':
            result = np.divide(result, count, out=np.zeros_like(result), where=count > 0)
        else:
            result *= meta['spacing'][dim]
    else:
        raise InputError('投影方式不支持')
    result = np.where(count > 0, result, 0)
    return {'images': {'projection': result, 'support': count}, 'report': {'axis': axis, 'range': [start, end], 'method': method, 'physical_depth_mm': (end - start) * meta['spacing'][dim], 'complete_rays': int((count == end - start).sum()), 'unsupported_rays': int((count == 0).sum()), 'note': '正交投影；积分仅累加有效支持，缺失的射线另见support，不代表完整透射投影。'}}


def crop(data, valid, meta, bounds):
    if not isinstance(bounds, list) or len(bounds) != 6:
        raise InputError('裁剪范围按[z0,z1,y0,y1,x0,x1]给出')
    ranges = []
    origin = []
    for axis in range(3):
        lo = integer(bounds[axis * 2], '裁剪起点', 0, data.shape[axis] - 2)
        hi = integer(bounds[axis * 2 + 1], '裁剪终点', lo + 2, data.shape[axis])
        ranges.append(slice(lo, hi))
        origin.append(meta['origin'][axis] + lo * meta['spacing'][axis])
    new_meta = dict(meta, origin=origin)
    return {'volume': data[tuple(ranges)].copy(), 'valid': valid[tuple(ranges)].copy(), 'meta': new_meta, 'report': {'bounds': bounds, 'origin_mm': origin, 'shape': list(data[tuple(ranges)].shape), 'coordinates': '体素中心；Z/Y/X顺序，右端范围不含'}}


def gaussian(data, valid, meta, sigma_mm=1, minimum_support=.75):
    sigma_mm = number(sigma_mm, '平滑半径mm', .001, 20)
    minimum_support = number(minimum_support, '最低支持', .01, 1)
    sigma = np.asarray([sigma_mm / s for s in meta['spacing']])
    if np.max(sigma) > 64:
        raise InputError('物理半径相对体素过大')
    weight = ndimage.gaussian_filter(valid.astype(np.float32), sigma=sigma, mode='constant')
    numerator = ndimage.gaussian_filter(np.where(valid, data, 0), sigma=sigma, mode='constant')
    result = np.divide(numerator, weight, out=np.zeros_like(numerator), where=weight > 1e-8)
    support = (weight >= minimum_support) & valid
    return {'volume': result, 'valid': support, 'maps': {'filter_support': weight}, 'report': {'sigma_mm': sigma_mm, 'sigma_voxels': sigma.tolist(), 'valid_before': int(valid.sum()), 'valid_after': int(support.sum()), 'minimum_support': minimum_support}}


def gradient(data, valid, meta):
    gradients = np.gradient(data.astype(np.float64), *meta['spacing'])
    full = ndimage.minimum_filter(valid.astype(np.uint8), size=3, mode='constant', cval=0).astype(bool)
    magnitude = np.sqrt(sum(a * a for a in gradients))
    return {'volume': np.where(full, magnitude, 0), 'valid': full, 'meta': dict(meta, unit=meta['unit'] + '/mm'), 'maps': {name: np.where(full, a, 0) for name, a in zip(['dz', 'dy', 'dx'], gradients)}, 'report': summary_stats(magnitude, full)}


def profiles(data, valid, meta):
    tables = {}
    for axis, name in enumerate(('z', 'y', 'x')):
        rows = []
        for index in range(data.shape[axis]):
            a, m = section(data, valid, name, index)
            stats = summary_stats(a, m)
            rows.append({'index': index, 'position_mm': meta['origin'][axis] + index * meta['spacing'][axis], 'valid_voxels': stats['valid_voxels'], 'mean': stats['mean'], 'minimum': stats['minimum'], 'maximum': stats['maximum'], 'standard_deviation': stats['standard_deviation']})
        tables[name + '_profile'] = rows
    return {'tables': tables, 'report': {'spacing': meta['spacing'], 'origin': meta['origin'], 'unit': meta['unit']}}


def histogram(data, valid, bins=64):
    bins = integer(bins, '直方图分箱', 8, 512)
    values = data[valid]
    if not values.size:
        raise InputError('没有有效体素')
    counts, edges = np.histogram(values, bins=bins)
    rows = [{'low': float(edges[i]), 'high': float(edges[i + 1]), 'count': int(c)} for i, c in enumerate(counts)]
    return {'tables': {'histogram': rows}, 'report': summary_stats(data, valid)}


def resample(data, valid, meta, spacing, mode='linear'):
    spacing = vector(spacing, '新体素间距', 3, .00001, 1000)
    if mode not in ('linear', 'nearest'):
        raise InputError('重采样方式无效')
    physical = (np.asarray(data.shape) - 1) * meta['spacing']
    shape = np.floor(physical / spacing + 1e-7).astype(int) + 1
    if np.min(shape) < 2 or np.max(shape) > 1024 or np.prod(shape) > 8_000_000:
        raise InputError('重采样网格超出容量')
    matrix = np.diag(np.asarray(spacing) / meta['spacing'])
    order = 1 if mode == 'linear' else 0
    support = ndimage.affine_transform(valid.astype(np.float32), matrix, output_shape=tuple(shape), order=order, mode='constant', cval=0, prefilter=False)
    numerator = ndimage.affine_transform(np.where(valid, data, 0), matrix, output_shape=tuple(shape), order=order, mode='constant', cval=0, prefilter=False)
    good = support >= 1 - 1e-6
    out = np.divide(numerator, support, out=np.zeros_like(numerator), where=support > 1e-6)
    return {'volume': out, 'valid': good, 'meta': dict(meta, spacing=spacing), 'maps': {'sampling_support': support}, 'report': {'old_shape': list(data.shape), 'new_shape': shape.tolist(), 'spacing': spacing, 'origin': meta['origin'], 'mode': mode, 'grid_note': '保持首体素中心与新间距；末中心不越过原末中心，可能有不足一新体素的尾部未采样。'}}


def compare(first, first_valid, first_meta, second, second_valid, second_meta):
    if first.shape != second.shape or not np.allclose(first_meta['spacing'], second_meta['spacing'], atol=1e-8, rtol=0) or not np.allclose(first_meta['origin'], second_meta['origin'], atol=1e-8, rtol=0):
        raise InputError('逐体素比较需要形状、间距和原点完全相同；此操作不进行配准')
    if first_meta['unit'] != second_meta['unit']:
        raise InputError('比较数据单位必须一致')
    good = first_valid & second_valid
    difference = second - first
    values = difference[good]
    report = summary_stats(difference, good)
    report['rmse'] = float(np.sqrt(np.mean(values ** 2))) if values.size else None
    report['mean_absolute_difference'] = float(np.mean(abs(values))) if values.size else None
    return {'volume': np.where(good, difference, 0), 'valid': good, 'report': report}


def sample_line(data, valid, meta, start, end, samples=128):
    start = np.asarray(vector(start, '起点Z/Y/X', 3))
    end = np.asarray(vector(end, '终点Z/Y/X', 3))
    samples = integer(samples, '样本数', 2, 2048)
    if np.allclose(start, end):
        raise InputError('剖线两端不能重合')
    coordinates = np.linspace(start, end, samples).T
    support = ndimage.map_coordinates(valid.astype(np.float32), coordinates, order=1, mode='constant', cval=0, prefilter=False)
    values = ndimage.map_coordinates(np.where(valid, data, 0), coordinates, order=1, mode='constant', cval=0, prefilter=False)
    length = float(np.linalg.norm((end - start) * meta['spacing']))
    rows = []
    for i in range(samples):
        good = support[i] >= 1 - 1e-6
        rows.append({'index': i, 'distance_mm': length * i / (samples - 1), 'z': float(coordinates[0, i]), 'y': float(coordinates[1, i]), 'x': float(coordinates[2, i]), 'valid': bool(good), 'value': float(values[i] / support[i]) if good else None})
    return {'tables': {'line': rows}, 'report': {'length_mm': length, 'valid_samples': sum(row['valid'] for row in rows), 'coordinates': '输入是体素中心索引；物理距离使用Z/Y/X间距'}}
