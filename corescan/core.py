import io
import json
import zipfile
import re
import numpy as np
from PIL import Image

MAX_VOXELS = 8_000_000
MAX_BYTES = 96 * 1024 * 1024


class InputError(ValueError):
    pass


def number(value, label, minimum=None, maximum=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InputError(label + '必须是数字')
    if not np.isfinite(result):
        raise InputError(label + '必须有限')
    if minimum is not None and result < minimum:
        raise InputError(label + '太小')
    if maximum is not None and result > maximum:
        raise InputError(label + '太大')
    return result


def integer(value, label, minimum, maximum):
    result = number(value, label, minimum, maximum)
    if result != int(result):
        raise InputError(label + '必须是整数')
    return int(result)


def vector(value, label, size, minimum=None, maximum=None):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise InputError(label + '长度不正确')
    return [number(item, label, minimum, maximum) for item in value]


def volume(data):
    arr = np.asarray(data)
    if arr.dtype.kind not in 'fiu' or arr.ndim != 3:
        raise InputError('需要 Z、Y、X 顺序的三维数值数组')
    if min(arr.shape) < 2 or max(arr.shape) > 1024 or arr.size > MAX_VOXELS:
        raise InputError('每轴须为2–1024且总体素不超过800万')
    if not np.isfinite(arr).all():
        raise InputError('输入体数据不能包含NaN或无穷值')
    if np.max(np.abs(arr.astype(np.float64))) > np.finfo(np.float32).max:
        raise InputError('体数据数值超出float32存储范围')
    return np.asarray(arr, dtype=np.float32)


def validity(value, shape):
    if value is None:
        return np.ones(shape, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != tuple(shape):
        raise InputError('有效支持形状不匹配')
    return arr


def metadata(value=None):
    value = value or {}
    if not isinstance(value, dict):
        raise InputError('体数据元信息需要对象')
    return {
        'spacing': vector(value.get('spacing', [1, 1, 1]), 'Z/Y/X体素间距', 3, .00001, 1000),
        'origin': vector(value.get('origin', [0, 0, 0]), 'Z/Y/X物理原点', 3, -1e9, 1e9),
        'unit': str(value.get('unit', 'attenuation'))[:40],
        'length_unit': 'mm',
        'description': str(value.get('description', ''))[:1000],
    }


def npy_bytes(data):
    output = io.BytesIO()
    np.save(output, np.asarray(data), allow_pickle=False)
    return output.getvalue()


def decode(raw, name):
    if len(raw) > MAX_BYTES:
        raise InputError('上传文件不能超过96MB')
    try:
        if name.lower().endswith('.npy'):
            return volume(np.load(io.BytesIO(raw), allow_pickle=False)), {}, None
        if name.lower().endswith('.npz'):
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                if 'volume' not in archive:
                    raise InputError('NPZ需要volume数组')
                data = volume(archive['volume'])
                meta = json.loads(str(archive['metadata_json'].item())) if 'metadata_json' in archive else {}
                for key in ('spacing', 'origin'):
                    if key in archive:
                        meta[key] = archive[key].tolist()
                good = validity(archive['valid'], data.shape) if 'valid' in archive else None
                return data, metadata(meta), good
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if sum(item.file_size for item in members) > MAX_BYTES or len(members) > 1025:
                    raise InputError('解压后超过容量预算')
                names = [item.filename for item in members if item.filename.lower().endswith(('.png', '.tif', '.tiff'))]
                if len(names) < 2:
                    raise InputError('ZIP中至少需要两张灰度切片')
                natural = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
                names.sort(key=natural)
                slices = []
                for path in names:
                    with Image.open(io.BytesIO(archive.read(path))) as image:
                        arr = np.asarray(image)
                    if arr.ndim != 2:
                        raise InputError('ZIP切片必须是单通道灰度图')
                    if slices and arr.shape != slices[0].shape:
                        raise InputError('所有切片尺寸必须相同')
                    slices.append(arr)
                    if sum(a.size for a in slices) > MAX_VOXELS:
                        raise InputError('切片总体素超过800万')
                meta = json.loads(archive.read('metadata.json')) if 'metadata.json' in archive.namelist() else {}
                return volume(np.stack(slices)), metadata(meta), None
    except InputError:
        raise
    except Exception as error:
        raise InputError('无法解码体数据：' + str(error)) from error
    raise InputError('支持NPY、含volume的NPZ或灰度切片ZIP')


def display(data, valid=None, low=None, high=None):
    data = np.asarray(data, dtype=np.float32)
    good = np.ones(data.shape, bool) if valid is None else np.asarray(valid, bool)
    finite = data[good & np.isfinite(data)]
    if not finite.size:
        return np.zeros(data.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1)) if low is None else number(low, '窗下限')
    hi = float(np.percentile(finite, 99)) if high is None else number(high, '窗上限')
    if hi <= lo:
        hi = lo + 1
    result = np.clip((data - lo) / (hi - lo), 0, 1)
    return np.where(good, result * 255, 0).astype(np.uint8)


def png_bytes(data, valid=None, low=None, high=None, labels=False):
    if labels:
        arr = np.asarray(data, dtype=np.int64)
        rgb = np.stack([(arr * 67) % 220 + 35, (arr * 131) % 220 + 35, (arr * 29) % 220 + 35], axis=-1).astype(np.uint8)
        rgb[arr == 0] = 0
        if valid is not None:
            rgb[~np.asarray(valid, bool)] = [45, 45, 45]
        image = Image.fromarray(rgb)
    else:
        image = Image.fromarray(display(data, valid, low, high))
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def axis_index(value):
    if value not in ('z', 'y', 'x'):
        raise InputError('轴必须为z、y或x')
    return ('z', 'y', 'x').index(value)


def summary_stats(data, valid):
    values = np.asarray(data)[valid]
    if not values.size:
        return {'valid_voxels': 0, 'total_voxels': int(data.size), 'minimum': None, 'maximum': None, 'mean': None, 'standard_deviation': None}
    return {'valid_voxels': int(values.size), 'total_voxels': int(data.size), 'minimum': float(values.min()), 'maximum': float(values.max()), 'mean': float(values.mean()), 'standard_deviation': float(values.std()), 'percentiles': dict(zip(['p01', 'p10', 'p50', 'p90', 'p99'], np.percentile(values, [1, 10, 50, 90, 99]).tolist()))}
