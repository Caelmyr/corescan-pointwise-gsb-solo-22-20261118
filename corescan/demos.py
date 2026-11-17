import numpy as np
from scipy import ndimage


def make_demo():
    shape = (48, 80, 88)
    spacing = [.55, .3, .3]
    origin = [-12.925, -11.85, -13.05]
    z, y, x = np.meshgrid(*[np.arange(n) * s + o for n, s, o in zip(shape, spacing, origin)], indexing='ij')
    rng = np.random.default_rng(23011)
    body = (x / 10) ** 2 + (y / 9) ** 2 < 1
    body &= abs(z) < 11
    hole = (x - 1.2 * np.sin(z / 7)) ** 2 + (y - .7 * np.cos(z / 6)) ** 2 < 2.1 ** 2
    solid = body & ~hole
    pores = np.zeros(shape, dtype=bool)
    for center, radii in [((-6, 4.3, 4), (1.2, 1, .8)), ((2.3, -5, 3), (1, 1.4, 1.1)), ((6.8, 3.2, -4.8), (.9, .8, 1.3))]:
        ellipsoid = ((z - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2 + ((x - center[2]) / radii[2]) ** 2 < 1
        pores |= ellipsoid
    crack = (abs(x + .25 * z + 5) < .25) & (abs(y) < 2.5) & (z > -3) & (z < 5)
    solid &= ~pores & ~crack
    density = np.where(solid, 1.4 + .06 * z / 11, .02)
    density = ndimage.gaussian_filter(density, [.35, .65, .65])
    density += rng.normal(0, .012, shape)
    scanned_again = ndimage.shift(density, [0.6, -1.1, .7], order=1, mode='constant', cval=.02)
    scanned_again += rng.normal(0, .009, shape)
    field = .03 * (x / 13) ** 2
    density += field
    meta = {'spacing': spacing, 'origin': origin, 'unit': 'attenuation', 'description': '合成工业零件；不是医学数据，也不用于安全认证。'}
    return [{'name': '合成铸件 · 原始扫描', 'data': density.astype(np.float32), 'meta': meta}, {'name': '合成铸件 · 重复扫描', 'data': scanned_again.astype(np.float32), 'meta': meta}], {'synthetic': True, 'shape': list(shape), 'description': '确定性生成的含弯曲孔道、3个椭圆孔隙和狭缝的零件，用于体验体数据查看、分区和测量；重复扫描包含轻微位置变化。'}
