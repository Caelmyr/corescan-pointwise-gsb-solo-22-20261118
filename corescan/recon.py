"""平行束透射投影的校正、重建与证据评估。

几何约定（全部在毫米单位下）：
- 投影栈形状为 (视图 V, 探测器行 R, 探测器列 C)；行对应相同轴向位置。
- 探测器列方向 u，像素间距 du；旋转中心在探测器列索引 center（可为小数）。
- 角度表以度给出，按 rotation_direction 取符号后转为弧度；重复、缺失角度都允许。
- 重建网格 N x N，间距 dx（默认等于 du），轴向间距 dz 由探测器行距给出。
无效射线（饱和、非正透射、截断边界、整图剔除）只进入支持掩码，绝不按零吸收处理。
"""
import io
import csv
import re
import zipfile
import numpy as np
from PIL import Image
from scipy import ndimage
from .core import InputError, number, integer

MAX_VIEWS = 240
MAX_SIDE = 320
MAX_RAYS = 26_000_000


def natural_key(value):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(value))]


def read_image(raw, name):
    try:
        with Image.open(io.BytesIO(raw)) as image:
            arr = np.asarray(image)
    except Exception as error:
        raise InputError('无法解码图像 ' + name + '：' + str(error)) from error
    if arr.ndim != 2:
        raise InputError(name + ' 必须是单通道二维灰度图')
    return arr.astype(np.float64)


def load_bundle(zip_bytes, angle_bytes, angle_name):
    """读取含 projections/ dark/ flat/ 的 ZIP 与角度表 CSV。"""
    angle_rows = parse_angle_table(angle_bytes, angle_name)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = [i for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith(('.png', '.tif', '.tiff'))]
        if len(members) < 3:
            raise InputError('投影压缩包至少需要两张投影和参考场图像')
        if sum(i.file_size for i in members) > 96 * 1024 * 1024 or len(members) > MAX_VIEWS + 4:
            raise InputError('投影数据超出容量预算（视图最多 ' + str(MAX_VIEWS) + '）')
        groups = {'projection': [], 'dark': [], 'flat': []}
        for info in members:
            parts = [p.lower() for p in info.filename.split('/')[:-1]]
            kind = next((k for k in groups if k in parts), 'projection')
            groups[kind].append(info)
        for kind in groups:
            groups[kind].sort(key=lambda i: natural_key(i.filename))
        if len(groups['projection']) < 2:
            raise InputError('压缩包内未找到至少两张投影（请放入 projections/ 目录）')
        images = {}
        shape = None
        for kind, infos in groups.items():
            for info in infos:
                arr = read_image(archive.read(info), info.filename)
                if shape is None:
                    shape = arr.shape
                elif arr.shape != shape:
                    raise InputError('所有投影、暗场与空场必须同尺寸；' + info.filename + ' 尺寸不一致')
                images[info.filename] = arr
    rows, cols = shape
    if rows < 4 or cols < 4 or rows > MAX_SIDE or cols > MAX_SIDE:
        raise InputError('探测器每边尺寸须在 4–' + str(MAX_SIDE) + ' 之间')
    if len(groups['projection']) * rows * cols > MAX_RAYS:
        raise InputError('投影射线总数超出预算，请减小尺寸或视图数')
    dark = _reference_field(groups['dark'], images)
    flat = _reference_field(groups['flat'], images)
    if dark is None or flat is None:
        raise InputError('压缩包需要 dark/ 与 flat/ 目录各含至少一张同尺寸参考场')
    names = [i.filename for i in groups['projection']]
    raw = np.stack([images[n] for n in names]).astype(np.float32)
    paired = []
    used = set()
    table = {row['filename']: row['angle'] for row in angle_rows}
    for name in names:
        base = name.split('/')[-1]
        match = next((key for key in table if key in (name, base) or name.endswith(key) or base.endswith(key)), None)
        if match is None:
            paired.append(None)
        else:
            used.add(match)
            paired.append(table[match])
    unmatched = [row['filename'] for row in angle_rows if row['filename'] not in used]
    return {'raw': raw, 'names': names, 'dark': dark.astype(np.float32), 'flat': flat.astype(np.float32),
            'angles': paired, 'unmatched_table_rows': unmatched,
            'angle_rows': angle_rows, 'shape': [rows, cols]}


def _reference_field(infos, images):
    if not infos:
        return None
    stacked = np.stack([images[i.filename] for i in infos])
    return np.median(stacked, axis=0)


def parse_angle_table(raw, name):
    text = raw.decode('utf-8-sig', errors='replace')
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        raise InputError('角度表 ' + name + ' 为空')
    start = 1 if any(h.strip().lower() in ('filename', 'file', 'name', '文件', '文件名') for h in rows[0]) else 0
    out = []
    for row in rows[start:]:
        if len(row) < 2:
            continue
        fname = row[0].strip()
        try:
            angle = float(row[1])
        except ValueError:
            raise InputError('角度表中 ' + fname + ' 的角度不是数字')
        if not np.isfinite(angle):
            raise InputError('角度表中 ' + fname + ' 的角度必须有限')
        out.append({'filename': fname, 'angle': angle})
    if not out:
        raise InputError('角度表未给出任何 文件名,角度(度) 行')
    return out


def correct_fields(raw, dark, flat, saturation=None, transmission_floor=1e-4, truncate_border=0):
    """暗场/空场校正与线积分转换；返回 (线积分, 有效射线掩码, 统计)。"""
    raw = np.asarray(raw, dtype=np.float64)
    dark = np.asarray(dark, dtype=np.float64)
    flat = np.asarray(flat, dtype=np.float64)
    if raw.shape[1:] != dark.shape or dark.shape != flat.shape:
        raise InputError('暗场、空场与投影尺寸不一致')
    denom = flat - dark
    if not np.isfinite(denom).all() or np.nanmedian(denom) <= 0:
        raise InputError('空场减暗场没有有效正信号，无法校正')
    if saturation is None:
        saturation = float(raw.max())
    saturation = number(saturation, '饱和阈值', 1e-9, 1e18)
    floor = number(transmission_floor, '透射率下限', 1e-12, 0.5)
    border = integer(truncate_border, '截断边界列数', 0, max(0, raw.shape[2] // 2 - 2))
    good = np.isfinite(raw) & np.isfinite(dark) & np.isfinite(flat) & (denom > 1e-12)
    good &= raw < saturation
    transmission = np.zeros_like(raw)
    np.divide(raw - dark, denom, out=transmission, where=good)
    good &= transmission > floor
    if border:
        good[:, :, :border] = False
        good[:, :, -border:] = False
    line = np.where(good, -np.log(np.clip(transmission, floor, None)), 0.0).astype(np.float32)
    stats = [_view_stats(line[v], good[v], transmission[v], raw[v], saturation) for v in range(raw.shape[0])]
    return line, good, {'saturation': saturation, 'transmission_floor': floor, 'truncate_border': border,
                        'valid_rays': int(good.sum()), 'total_rays': int(good.size), 'views': stats}


def _view_stats(line, good, transmission, raw, saturation):
    vals = line[good]
    edge = max(1, good.shape[1] // 10)
    edge_atten = 1.0 - transmission[:, [0, -1]]
    return {'valid_rays': int(good.sum()), 'fraction_valid': float(good.mean()),
            'saturated_rays': int((raw >= saturation).sum()),
            'mean_line_integral': float(vals.mean()) if vals.size else None,
            'max_line_integral': float(vals.max()) if vals.size else None,
            'edge_attenuation_mean': float(np.clip(edge_atten, 0, None).mean())}


class ParallelBeam:
    """基于 map_coordinates 的平行束正/反投影；反投影带角度中点权重。"""

    def __init__(self, angles_rad, center, du_mm, grid_n, dx_mm, rows):
        self.angles = np.asarray(angles_rad, dtype=np.float64)
        self.center = float(center)
        self.du = float(du_mm)
        self.n = int(grid_n)
        self.dx = float(dx_mm)
        self.rows = int(rows)
        self.weights = angular_weights(self.angles)
        axis = (np.arange(self.n) - (self.n - 1) / 2.0) * self.dx
        self.xx, self.yy = np.meshgrid(axis, axis)
        self.v_index = (np.arange(self.n) - (self.n - 1) / 2.0) * self.dx
        det_u = (np.arange(self.n) - self.center) * self.du
        self.ui_grid, self.vi_grid = np.meshgrid(det_u, self.v_index)
        self.z_axis = np.arange(self.rows)
        self._det_count = max(4, int(round((self.n * self.dx) / self.du)))

    def forward(self, volume):
        volume = np.asarray(volume, dtype=np.float64)
        n_det = len(self._detector_axis())
        out = np.zeros((len(self.angles), self.rows, n_det))
        for v, theta in enumerate(self.angles):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x = self.ui_grid * cos_t - self.vi_grid * sin_t
            y = self.ui_grid * sin_t + self.vi_grid * cos_t
            ix = x / self.dx + (self.n - 1) / 2.0
            iy = y / self.dx + (self.n - 1) / 2.0
            zc, iyc, ixc = np.broadcast_arrays(self.z_axis[:, None, None], iy[None], ix[None])
            rotated = ndimage.map_coordinates(volume, [zc, iyc, ixc], order=1, mode='constant', cval=0.0, prefilter=False)
            ray = rotated.sum(axis=1) * self.dx
            det = self._detector_axis()
            target_u = (det - self.center) * self.du
            source_u = (np.arange(self.n) - self.center) * self.du
            source_index = np.interp(target_u, source_u, np.arange(self.n), left=np.nan, right=np.nan)
            rows_index, _ = np.meshgrid(self.z_axis, source_index, indexing='ij')
            sampled = ndimage.map_coordinates(ray, [rows_index, np.broadcast_to(source_index, rows_index.shape)],
                                              order=1, mode='constant', cval=0.0, prefilter=False)
            out[v] = sampled
        return out.astype(np.float32)

    def back(self, sinogram):
        """伴随反投影（含角度中点权重），用于迭代最小二乘。"""
        sinogram = np.asarray(sinogram, dtype=np.float64)
        out = np.zeros((self.rows, self.n, self.n))
        for v, theta in enumerate(self.angles):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            u = self.xx * cos_t + self.yy * sin_t
            d = u / self.du + self.center
            zc, dc = np.broadcast_arrays(self.z_axis[:, None, None], d[None])
            sampled = ndimage.map_coordinates(sinogram[v], [zc, dc], order=1, mode='constant', cval=0.0, prefilter=False)
            out += sampled * self.weights[v]
        return out.astype(np.float32)

    def back_plain(self, sinogram):
        """不带角度权重的逐视图求和反投影，供 FBP 自行控制角度归一化。"""
        sinogram = np.asarray(sinogram, dtype=np.float64)
        out = np.zeros((self.rows, self.n, self.n))
        for v, theta in enumerate(self.angles):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            u = self.xx * cos_t + self.yy * sin_t
            d = u / self.du + self.center
            zc, dc = np.broadcast_arrays(self.z_axis[:, None, None], d[None])
            sampled = ndimage.map_coordinates(sinogram[v], [zc, dc], order=1, mode='constant', cval=0.0, prefilter=False)
            out += sampled
        return out.astype(np.float32)

    def _detector_axis(self):
        return np.arange(self._det_count)


def angular_weights(angles):
    """角度中点权重（弧度）；同角重复视图平分该角权重，缺角自然形成大间隔。

    唯一角按 0..2π 排序：内角权重为相邻间隔各半，首尾两角只取内侧半个间隔，
    避免把圆周接缝间隔重复计入（全 0..π 等间隔扫描时权重和恰为 π）。
    """
    angles = np.mod(np.asarray(angles, dtype=np.float64), 2 * np.pi)
    unique, inverse, counts = np.unique(np.round(angles, 10), return_inverse=True, return_counts=True)
    gaps = np.diff(np.concatenate([unique, unique[:1] + 2 * np.pi]))
    group_weight = np.empty(len(unique))
    if len(unique) == 1:
        group_weight[0] = np.pi
    else:
        group_weight[0] = gaps[0] / 2.0
        group_weight[-1] = gaps[-2] / 2.0
        group_weight[1:-1] = gaps[:-2] / 2.0 + gaps[1:-1] / 2.0
    return (group_weight[inverse] / counts[inverse]).astype(np.float64)


def ramp_kernel(num_det, du, window='hann'):
    """频域斜坡 |q|（q=fftfreq，单位 1/mm）乘以可选 Hann 窗。

    最终角度归一化在 fbp_reconstruct 中按角度中点权重处理，本函数只负责频率整形。
    """
    if window not in ('hann', 'none'):
        raise InputError('窗函数只支持 hann 或 none')
    q = np.fft.fftfreq(num_det, d=du)
    ramp = np.abs(q).astype(np.float64)
    if window == 'hann':
        ramp *= 0.5 * (1 + np.cos(2 * np.pi * np.fft.fftfreq(num_det, d=1.0)))
    return ramp


def ramp_filter(num_det, du, window='hann'):
    return ramp_kernel(num_det, du, window)


def fbp_reconstruct(operator, sinogram, mask, window='hann'):
    """直接恢复：斜坡滤波 + 加权反投影。

    离散尺度标定（高斯体与冲激数值核验）：频域 |q|（fftfreq 约定，量纲 1/mm，
    已吸收探测器间距 du）。等间隔全角扫描时，滤波后逐视图等权反投影求和、再乘
    角度步长 Δθ（弧度）即恢复正确幅值，与 du、网格点数、角度密度均无关。
    非均匀角时逐视图改用角度中点权重 w_v，整体乘权重均值 mean(w) 作为等效 Δθ，
    重复角由权重平分处理，缺角自然拉大局部间隔。系数只由几何决定，与物体形状无关。
    注意必须用不带权重的 back_plain，否则角度权重被重复计入。
    """
    sinogram = np.where(mask, sinogram, 0.0)
    num_det = sinogram.shape[2]
    ramp = ramp_filter(num_det, operator.du, window)
    filtered = np.real(np.fft.ifft(np.fft.fft(sinogram, axis=2) * ramp[None, None, :], axis=2))
    dtheta = float(operator.weights.mean())
    weighted = filtered * (operator.weights[:, None, None] / dtheta)
    image = operator.back_plain(weighted) * dtheta
    return image.astype(np.float32), dtheta


def smooth_gradient(volume, lam):
    """二维四邻域拉普拉斯平滑项梯度（各轴向切片独立）。"""
    lap = np.zeros_like(volume)
    lap += 4 * volume
    lap -= np.roll(volume, 1, axis=1) + np.roll(volume, -1, axis=1)
    lap -= np.roll(volume, 1, axis=2) + np.roll(volume, -1, axis=2)
    return lam * lap


def _lipschitz(operator, mask, lam, iterations=4):
    probe = np.random.default_rng(7).standard_normal((operator.rows, operator.n, operator.n))
    norm = np.linalg.norm(probe) + 1e-12
    for _ in range(iterations):
        probe = probe / norm
        forward = operator.forward(probe)
        backward = operator.back(forward * mask)
        probe = backward + smooth_gradient(probe, lam)
        norm = np.linalg.norm(probe)
    return float(norm) + 1e-9


def fista_reconstruct(operator, sinogram, mask, init, lam, iterations):
    """带非负与平滑约束的逐步最小二乘拟合；只拟合有效射线。"""
    lam = number(lam, '平滑权重', 0, 1e6)
    iterations = integer(iterations, '迭代次数', 1, 60)
    step = 1.0 / _lipschitz(operator, mask, lam)
    x = np.clip(np.asarray(init, dtype=np.float64), 0, None)
    y = x.copy()
    momentum = 1.0
    objective = []
    for _ in range(iterations):
        residual = operator.forward(y) * mask - sinogram * mask
        grad = operator.back(residual) + smooth_gradient(y, lam)
        x_next = np.clip(y - step * grad, 0, None)
        momentum_next = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        y = x_next + ((momentum - 1) / momentum_next) * (x_next - x)
        x, momentum = x_next, momentum_next
        objective.append(float(0.5 * np.sum(residual ** 2) + 0.5 * lam * _smooth_energy(x)))
    return x.astype(np.float32), {'step': step, 'iterations': iterations, 'lambda': lam, 'objective': objective}


def _smooth_energy(volume):
    energy = np.zeros(volume.shape[0])
    for axis in (1, 2):
        diff = np.diff(volume, axis=axis)
        energy = energy + np.sum(diff ** 2, axis=(1, 2))
    return float(energy.sum())


def support_status(operator, ray_mask, view_present, truncate_border, gap_limit_deg=60.0, weak_limit_deg=120.0):
    """三级支持：2 可靠 / 1 未决 / 0 不可恢复。约束补出的结构不会出现在可靠区内。

    ray_mask 为逐射线有效掩码（含饱和/截断/整图剔除）；view_present 为逐视图存在标志，
    独立留出视图仍计入角覆盖——它们是真实采集到的视角，只是不参与拟合。
    """
    ray_mask = np.asarray(ray_mask, dtype=np.float64)
    presence = np.zeros_like(ray_mask)
    presence[np.asarray(view_present, bool)] = 1.0
    fraction = np.divide(operator.back(ray_mask), np.maximum(operator.back(presence), 1e-12))
    yy, xx = np.meshgrid((np.arange(operator.n) - (operator.n - 1) / 2) * operator.dx,
                         (np.arange(operator.n) - (operator.n - 1) / 2) * operator.dx, indexing='ij')
    radius_mm = np.sqrt(xx ** 2 + yy ** 2)
    fov_radius = (operator._det_count / 2.0 - truncate_border) * operator.du
    inside_fov = radius_mm <= fov_radius
    used_angles = np.mod(operator.angles[np.asarray(view_present, bool)], 2 * np.pi)
    if used_angles.size >= 2:
        ordered = np.sort(used_angles)
        gaps = np.diff(np.concatenate([ordered, ordered[:1] + 2 * np.pi]))
        largest_gap = float(np.degrees(gaps.max()))
        coverage = float((ordered[-1] - ordered[0]) if ordered.size > 1 else 0.0)
    else:
        largest_gap, coverage = 360.0, 0.0
    status = np.zeros((operator.rows, operator.n, operator.n), dtype=np.uint8)
    reliable = inside_fov & (fraction >= 0.85) & (largest_gap <= gap_limit_deg)
    uncertain = inside_fov & (fraction >= 0.30) & (largest_gap <= weak_limit_deg) & ~reliable
    status[uncertain] = 1
    status[reliable] = 2
    return {'status': status, 'ray_fraction': fraction.astype(np.float32), 'fov_radius_mm': float(fov_radius),
            'largest_gap_deg': largest_gap, 'angular_coverage_deg': float(np.degrees(coverage)),
            'reliable_voxels': int((status == 2).sum()), 'uncertain_voxels': int((status == 1).sum()),
            'unsupported_voxels': int((status == 0).sum())}


def residual_metrics(forward, sinogram, mask):
    residual = np.where(mask, forward - sinogram, 0.0).astype(np.float32)
    vals = residual[mask]
    if not vals.size:
        return {'rmse': None, 'mae': None, 'valid_rays': 0}, residual
    return {'rmse': float(np.sqrt(np.mean(vals ** 2))), 'mae': float(np.mean(np.abs(vals))),
            'max_abs': float(np.max(np.abs(vals))), 'valid_rays': int(vals.size)}, residual


def build_geometry(angles_deg, direction, center, du, dz, grid_n, dx, rows):
    direction = 1 if int(direction) >= 0 else -1
    du = number(du, '探测器像素间距mm', 1e-6, 1e4)
    dz = number(dz, '轴向间距mm', 1e-6, 1e4)
    center = number(center, '旋转中心(探测器列索引)', -1e6, 1e6)
    grid_n = integer(grid_n, '重建网格边长', 16, MAX_SIDE)
    dx = number(dx if dx is not None else du, '重建体素间距mm', 1e-6, 1e4)
    if grid_n ** 3 > 8_000_000:
        raise InputError('重建网格总体素超过800万预算')
    signed = [None if a is None else direction * np.radians(a) for a in angles_deg]
    if any(a is None for a in signed):
        raise InputError('仍有投影缺少角度配对，不能重建')
    return ParallelBeam(signed, center, du, grid_n, dx, rows), {'direction': direction, 'center': center,
            'du_mm': du, 'dz_mm': dz, 'grid_n': grid_n, 'dx_mm': dx,
            'origin_mm': [-(rows - 1) / 2 * dz, -(grid_n - 1) / 2 * dx, -(grid_n - 1) / 2 * dx]}


def reconstruct_pipeline(raw, dark, flat, angles_deg, excluded, settings):
    """一次完整双方案重建；holdout 视图只用于独立证据，不参与任何方案。"""
    line, mask, correction = correct_fields(raw, dark, flat, settings.get('saturation'),
                                            settings.get('transmission_floor', 1e-4),
                                            settings.get('truncate_border', 0))
    excluded = np.asarray(excluded, dtype=bool)
    if excluded.shape != (raw.shape[0],):
        raise InputError('剔除集合长度与视图数不符')
    mask &= ~excluded[:, None, None]
    holdout = np.asarray(settings.get('holdout', []), dtype=int)
    holdout_set = set(holdout.tolist())
    if holdout.size and (holdout.min() < 0 or holdout.max() >= raw.shape[0]):
        raise InputError('留出视图索引越界')
    training_mask = mask.copy()
    training_mask[holdout] = False
    holdout_only_mask = np.zeros_like(mask)
    if holdout.size:
        holdout_only_mask[holdout] = mask[holdout]
    operator, geometry = build_geometry(angles_deg, settings.get('direction', 1), settings.get('center'),
                                        settings.get('du_mm'), settings.get('dz_mm'), settings.get('grid_n'),
                                        settings.get('dx_mm'), raw.shape[1])
    direct, scale = fbp_reconstruct(operator, line, training_mask, settings.get('fbp_window', 'hann'))
    constrained, fit_info = fista_reconstruct(operator, line, training_mask, np.clip(direct, 0, None),
                                              settings.get('lambda_smooth', 0.05), settings.get('iterations', 15))
    support = support_status(operator, mask, ~excluded, correction['truncate_border'])
    schemes = {}
    residuals = {}
    for name, volume in (('direct', direct), ('constrained', constrained)):
        forward = operator.forward(volume)
        train_metrics, train_residual = residual_metrics(forward, line, training_mask)
        holdout_metrics, _ = residual_metrics(forward, line, holdout_only_mask)
        valid_metrics, all_valid_residual = residual_metrics(forward, line, mask)
        per_view = []
        for v in range(raw.shape[0]):
            if not mask[v].any():
                per_view.append({'view': v, 'role': 'excluded' if excluded[v] else 'no_valid_rays', 'rmse': None})
            elif v in holdout:
                m, _ = residual_metrics(forward[v:v + 1], line[v:v + 1], mask[v:v + 1])
                per_view.append({'view': v, 'role': 'holdout', 'rmse': m['rmse'], 'mae': m['mae']})
            else:
                m, _ = residual_metrics(forward[v:v + 1], line[v:v + 1], training_mask[v:v + 1])
                per_view.append({'view': v, 'role': 'training', 'rmse': m['rmse'], 'mae': m['mae']})
        schemes[name] = {'train': train_metrics, 'holdout': holdout_metrics,
                         'all_valid': valid_metrics, 'per_view': per_view}
        residuals[name] = {'train': train_residual, 'all_valid': all_valid_residual, 'forward': forward}
    difference = constrained - direct
    return {'line': line, 'mask': mask, 'training_mask': training_mask, 'operator': operator,
            'geometry': geometry, 'correction': correction, 'support': support,
            'volumes': {'direct': direct, 'constrained': constrained},
            'metrics': schemes, 'residuals': residuals, 'difference': difference,
            'fbp_scale': scale, 'fit_info': fit_info, 'excluded': excluded, 'holdout': holdout}


def reproject(volume, angles_deg, settings, line, mask, excluded, holdout):
    """把恢复体按原始角度、方向和顺序重新投影，并在同样的有效射线上给残差。"""
    operator, geometry = build_geometry(angles_deg, settings.get('direction', 1), settings.get('center'),
                                        settings.get('du_mm'), settings.get('dz_mm'), settings.get('grid_n'),
                                        settings.get('dx_mm'), volume.shape[0])
    forward = operator.forward(volume)
    metrics, residual = residual_metrics(forward, line, mask)
    per_view = []
    for v in range(forward.shape[0]):
        role = 'excluded' if excluded[v] else ('holdout' if v in np.asarray(holdout, dtype=int) else 'training')
        if mask[v].any():
            m, _ = residual_metrics(forward[v:v + 1], line[v:v + 1], mask[v:v + 1])
            per_view.append({'view': v, 'role': role, 'rmse': m['rmse'], 'mae': m['mae']})
        else:
            per_view.append({'view': v, 'role': role, 'rmse': None, 'mae': None})
    return {'forward': forward, 'residual': residual, 'metrics': metrics, 'per_view': per_view,
            'ordering_note': '视图顺序、旋转方向与原始投影一致；缺失/剔除支持仅出现在掩码中，残差不把它们当零吸收。'}
