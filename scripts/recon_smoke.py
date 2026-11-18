#!/usr/bin/env python3
"""端到端冒烟测试：合成已知零件 → 平行束投影（含坏点/截断/重复缺角）→ 双方案重建 → 证据与往返。

不依赖服务器；在临时目录生成合成素材，验证：
1. 暗空场校正不把饱和/缺测当零吸收；2. FBP 与约束拟合同已知零件的一致性；
3. 独立留出视图只作证据；4. 支持图把缺角/截断区标为未决或不可恢复；
5. 会话式 API（Flask 测试客户端）的冻结、确认、克隆与导出。
"""
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

# -I 隔离模式下 PYTHONPATH 不自动进入 sys.path；这里仅加入受信解释器包目录，
# 合成素材始终位于临时目录并以路径参数传入，不会进入模块搜索路径。
for _p in os.environ.get('PYTHONPATH', '').split(os.pathsep):
    if _p and _p not in sys.path:
        sys.path.append(_p)

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corescan import recon as rc
from corescan.recon_server import install_recon
from corescan.storage import Store
from flask import Flask


def make_phantom(n=48, rows=16):
    """物理可透射的薄壁零件：衰减 0.02–0.12，最长路径线积分仍高于噪声本底。"""
    axis = (np.arange(n) - (n - 1) / 2)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    r = np.sqrt(xx ** 2 + yy ** 2)
    slice2d = np.where(r < n * 0.36, 0.08, 0.0)
    hole = ((xx - 5) ** 2 + (yy + 3) ** 2) < 3.2 ** 2
    wall = (r > n * 0.22) & (r < n * 0.28)
    slice2d[hole] = 0.0
    slice2d[wall] = 0.12
    volume = np.broadcast_to(slice2d[None], (rows, n, n)).copy().astype(np.float32)
    return volume


def make_projections(volume, du=1.0, dz=1.0):
    rows, n, _ = volume.shape
    angles_deg = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330, 30]
    op = rc.ParallelBeam([np.radians(a) for a in angles_deg], center=(n - 1) / 2, du_mm=du, grid_n=n, dx_mm=du, rows=rows)
    lines = op.forward(volume)
    transmission = np.exp(-lines)
    flat = np.full((rows, n), 1000.0)
    dark = np.full((rows, n), 10.0)
    raw = dark + (flat - dark) * transmission
    raw = np.clip(raw, 0, 990)
    raw[2, 4:7, 10:13] = 990.0
    raw[7, :, :2] = 990.0
    rng = np.random.default_rng(9)
    raw = np.clip(raw + rng.normal(0, 2.0, raw.shape), 0, 990)
    return raw.astype(np.float32), dark.astype(np.float32), flat.astype(np.float32), angles_deg


def write_bundle(tmp, raw, dark, flat, angles_deg):
    data_dir = tmp / 'bundle'
    for sub in ('projections', 'dark', 'flat'):
        (data_dir / sub).mkdir(parents=True)
    names = []
    for v in range(raw.shape[0]):
        name = 'projections/view_%03d.png' % v
        Image.fromarray(raw[v].astype(np.uint16)).save(data_dir / name)
        names.append(name.split('/')[-1])
    Image.fromarray(dark.astype(np.uint16)).save(data_dir / 'dark/dark_000.png')
    Image.fromarray(flat.astype(np.uint16)).save(data_dir / 'flat/flat_000.png')
    csv_text = 'filename,angle_deg\n' + '\n'.join('%s,%s' % (names[i], angles_deg[i]) for i in range(len(names)))
    (data_dir / 'angles.csv').write_text(csv_text)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as zf:
        for path in sorted(data_dir.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(data_dir))
    return buf.getvalue(), (data_dir / 'angles.csv').read_bytes()


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('  OK -', message)


def main():
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        print('1) 合成零件与含缺陷投影')
        volume = make_phantom()
        raw, dark, flat, angles = make_projections(volume)
        bundle_bytes, angle_bytes = write_bundle(tmp, raw, dark, flat, angles)
        bundle = rc.load_bundle(bundle_bytes, angle_bytes, 'angles.csv')
        check(bundle['raw'].shape == raw.shape, 'ZIP 解包形状一致 %s' % (bundle['raw'].shape,))
        check(all(a is not None for a in bundle['angles']), '每个投影都按文件名配对到角度')
        check(not bundle['unmatched_table_rows'], '角度表无未匹配行')

        print('2) 暗场空场校正与坏点/截断处理')
        line, mask, correction = rc.correct_fields(raw, dark, flat, saturation=985.0, truncate_border=1)
        check(mask.sum() < mask.size, '存在无效射线（饱和/截断被识别）')
        check(not mask[2, 4:7, 10:13].any(), '饱和坏点未被当作零吸收')
        check(not mask[7, :, :1].any() and not mask[7, :, -1:].any(), '截断边界列为无效')
        check(np.isfinite(line[mask]).all() and (line[mask] >= 0).all(), '有效射线上的线积分有限且非负')
        check(correction['views'][2]['saturated_rays'] >= 9, '视图级饱和统计正确')

        print('3) 双方案重建（含重复角、缺角与独立留出）')
        excluded = [False] * len(angles)
        excluded[11] = True
        settings = {'du_mm': 1.0, 'dz_mm': 1.0, 'center': 23.5, 'direction': 1,
                    "saturation": 985.0, 'transmission_floor': 1e-4, 'truncate_border': 1,
                    'grid_n': 48, 'dx_mm': 1.0, 'lambda_smooth': 0.002, 'iterations': 20, 'holdout': [5, 13]}
        outcome = rc.reconstruct_pipeline(raw, dark, flat, angles, excluded, settings)
        direct, constrained = outcome['volumes']['direct'], outcome['volumes']['constrained']
        reliable = outcome['support']['status'] == 2
        check(reliable.any(), '存在可靠支持体素')
        check(reliable.mean() > 0.2, '可靠支持覆盖中心区域 %.1f%%' % (100 * reliable.mean()))
        check((outcome['support']['status'] == 0).any(), '视场/角度不足区域被标记为不可恢复')
        check(outcome['support']['largest_gap_deg'] > 40, '缺角（剔除视图）形成可报告的最大角间隔 %.1f°' % outcome['support']['largest_gap_deg'])
        rmse_direct = float(np.sqrt(np.mean((direct[reliable] - volume[reliable]) ** 2)))
        rmse_constrained = float(np.sqrt(np.mean((constrained[reliable] - volume[reliable]) ** 2)))
        check(rmse_direct < 0.06, 'FBP 可靠区 RMSE=%.4f 恢复出已知零件量级（真值≤0.12）' % rmse_direct)
        check(rmse_constrained < 0.04, '约束拟合可靠区 RMSE=%.4f 接近已知零件' % rmse_constrained)
        check(rmse_constrained <= rmse_direct + 1e-9, '稀疏角下约束拟合可靠区不劣于直接恢复（其代价由独立证据另行判断）')
        check((constrained >= -1e-6).all(), '约束拟合处处非负')
        check(len(outcome['fit_info']['objective']) == 20 and np.all(np.diff(outcome['fit_info']['objective']) <= 1e-6), 'FISTA 目标值不回升')
        holdout_views = {row['view'] for row in outcome['metrics']['direct']['per_view'] if row['role'] == 'holdout'}
        check(holdout_views == {5, 13}, '留出视图被独立标记且未参与训练')
        for name in ('direct', 'constrained'):
            m = outcome['metrics'][name]
            check(m['holdout']['rmse'] is not None and np.isfinite(m['holdout']['rmse']),
                  name + ' 在未参与重建的投影上给出独立 RMSE=%.4f（误差低不自动代表更可信）' % m['holdout']['rmse'])
        check(outcome['metrics']['constrained']['holdout']['rmse'] < 0.15,
              '约束方案在独立留出投影上 RMSE=%.4f 与观测一致' % outcome['metrics']['constrained']['holdout']['rmse'])
        check(outcome['residuals']['direct']['train'].shape == line.shape, '训练残差形状与原始投影一致')
        check(not outcome['residuals']['direct']['train'][~outcome['training_mask']].any(), '残差在无效/留出射线上为零而非伪造零吸收')

        print('4) 往返重投影保持顺序、方向与支持')
        trip = rc.reproject(constrained, angles, settings, line, mask, np.array(excluded), np.array([5, 13]))
        check(trip['forward'].shape == raw.shape, '重投影形状与原始投影一致')
        check(trip['metrics']['rmse'] < 0.2, '往返有效射线 RMSE=%.4f' % trip['metrics']['rmse'])
        check(not trip['residual'][~mask].any(), '往返残差只落在有效射线上')
        roles = {r['view']: r['role'] for r in trip['per_view']}
        check(roles[11] == 'excluded' and roles[5] == 'holdout', '往返链上剔除/留出角色一致')

        print('5) 会话 API：冻结、确认、克隆、项目导出')
        app = Flask(__name__)
        store = Store(tmp / 'store')
        install_recon(app, store)
        client = app.test_client()
        created = client.post('/core/projects', json={'name': '冒烟'}).get_json()
        pid = created['id']
        with open(bundle_path := tmp / 'b.zip', 'wb') as fh:
            fh.write(bundle_bytes)
        with open(angle_path := tmp / 'a.csv', 'wb') as fh:
            fh.write(angle_bytes)
        with open(bundle_path, 'rb') as fb, open(angle_path, 'rb') as fa:
            resp = client.post('/core/projects/%s/recon/sessions' % pid, data={
                'bundle': (fb, 'bundle.zip'), 'angles': (fa, 'angles.csv'), 'revision': 0,
                'settings': json.dumps({k: v for k, v in settings.items() if k != 'holdout'})},
                content_type='multipart/form-data')
        check(resp.status_code == 200, '创建会话成功')
        doc = resp.get_json()['project']
        sid = doc['recon_sessions'][0]['id']
        rev = doc['revision']
        resp = client.post('/core/projects/%s/recon/sessions/%s/curation' % (pid, sid),
                           json={'revision': rev, 'excluded': excluded, 'holdout': [5, 13]})
        check(resp.status_code == 200, '剔除与留出保存成功')
        rev = resp.get_json()['project']['revision']
        resp = client.post('/core/projects/%s/recon/sessions/%s/reconstruct' % (pid, sid), json={'revision': rev})
        check(resp.status_code == 200, '双方案重建 API 成功')
        rev = resp.get_json()['project']['revision']
        frozen = client.post('/core/projects/%s/recon/sessions/%s/curation' % (pid, sid),
                             json={'revision': rev, 'settings': {'center': 10.0}})
        check(frozen.status_code == 409, '重建后设置被冻结，拒绝改写')
        resp = client.post('/core/projects/%s/recon/sessions/%s/confirm' % (pid, sid),
                           json={'revision': rev, 'scheme': 'constrained', 'reason': '冒烟测试裁决'})
        check(resp.status_code == 200, '确认体生成成功')
        vid = resp.get_json()['result']['volume_id']
        rev = resp.get_json()['project']['revision']
        again = client.post('/core/projects/%s/recon/sessions/%s/confirm' % (pid, sid),
                            json={'revision': rev, 'scheme': 'direct'})
        check(again.status_code == 409, '已确认会话不可再次确认（确认体不可变）')
        final_doc = store.load(pid)
        record = store.find(final_doc, vid)
        check(record['meta']['spacing'] == [1.0, 1.0, 1.0] and len(record['meta']['origin']) == 3, '确认体带毫米间距与原点')
        _, valid = store.arrays(record)
        check(valid.sum() < valid.size and valid.sum() > 0, '确认体支持掩码只保留可靠区')
        clone = client.post('/core/projects/%s/recon/sessions/%s/clone' % (pid, sid), json={'revision': final_doc['revision']})
        check(clone.status_code == 200 and clone.get_json()['result']['state'] == 'curating', '克隆会话回到可编辑状态且不动确认体')
        exported = client.get('/core/projects/%s/download' % pid)
        check(exported.status_code == 200 and b'project.json' in exported.data, '项目导出包含完整会话对象')
        with zipfile.ZipFile(io.BytesIO(exported.data)) as zf:
            keys = {n.split('/')[-1].replace('.npy', '') for n in zf.namelist() if n.startswith('objects/')}
        session_doc = next(s for s in store.load(pid)['recon_sessions'] if s['id'] == sid)
        needed = {k for group in session_doc['arrays'].values() for k in group.values()}
        check(needed <= keys, '导出对象覆盖会话全部内容寻址数组（%d 个）' % len(needed))
        image = client.get('/core/projects/%s/recon/sessions/%s/image/schemes/direct_volume?axis=z' % (pid, sid))
        check(image.status_code == 200 and image.data[:4] == b'\x89PNG', '重建体切片图像可查看')
        print('\n全部冒烟检查通过。')


if __name__ == '__main__':
    main()
