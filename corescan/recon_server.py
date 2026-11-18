"""从原始投影重建切片体的会话式 API。

会话状态：curating（核对/剔除/留出）→ reconstructed（双方案证据齐备，待裁决）
→ confirmed（已生成不可变的确认体；之后改角度或几何只能新建会话）。
所有输入与派生数组经内容寻址存储；项目导出一并带走，重开后证据链完整。
"""
import io
import csv
import json
import time
import uuid
import zipfile
import numpy as np
from flask import request, jsonify, send_file, Response, render_template
from PIL import Image
from .core import InputError, png_bytes, npy_bytes
from .storage import Conflict
from . import recon as rc

SETTING_FIELDS = ('du_mm', 'dz_mm', 'center', 'direction', 'saturation', 'transmission_floor',
                  'truncate_border', 'grid_n', 'dx_mm', 'lambda_smooth', 'iterations', 'fbp_window')
ARRAY_LAYOUT = ('inputs', 'corrected', 'support', 'schemes', 'comparison', 'roundtrip')


def install_recon(app, store):

    def body():
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            raise InputError('需要JSON对象')
        return value

    def find_session(doc, sid):
        session = next((s for s in doc.get('recon_sessions', []) if s['id'] == sid), None)
        if session is None:
            raise InputError('重建会话不存在')
        return session

    def put_arr(arr):
        arr = np.asarray(arr)
        if arr.dtype == bool:
            arr = arr.astype(np.uint8)
        return store.put(arr)

    def get_arr(session, group, name):
        key = session['arrays'].get(group, {}).get(name)
        if key is None:
            raise InputError('数组不存在或尚未生成')
        return store.array(key)

    def public(session):
        """返回不携带对象键以外冗余的会话摘要。"""
        summary = json.loads(json.dumps(session, ensure_ascii=False))
        summary['arrays'] = {g: sorted(names) for g, names in session['arrays'].items()}
        return summary

    def validate_settings(settings, bundle_shape):
        clean = {}
        if 'du_mm' in settings:
            clean['du_mm'] = rc.number(settings['du_mm'], '探测器像素间距mm', 1e-6, 1e4)
        if 'dz_mm' in settings:
            clean['dz_mm'] = rc.number(settings['dz_mm'], '轴向间距mm', 1e-6, 1e4)
        if 'center' in settings:
            clean['center'] = rc.number(settings['center'], '旋转中心(探测器列索引)', -1e6, 1e6)
        if 'direction' in settings:
            clean['direction'] = 1 if int(settings['direction']) >= 0 else -1
        if 'saturation' in settings:
            clean['saturation'] = rc.number(settings['saturation'], '饱和阈值', 1e-9, 1e18)
        if 'transmission_floor' in settings:
            clean['transmission_floor'] = rc.number(settings['transmission_floor'], '透射率下限', 1e-12, 0.5)
        if 'truncate_border' in settings:
            clean['truncate_border'] = rc.integer(settings['truncate_border'], '截断边界列数', 0, bundle_shape[1] // 2 - 2)
        if 'grid_n' in settings and settings['grid_n'] is not None:
            clean['grid_n'] = rc.integer(settings['grid_n'], '重建网格边长', 16, rc.MAX_SIDE)
        if 'dx_mm' in settings and settings['dx_mm'] is not None:
            clean['dx_mm'] = rc.number(settings['dx_mm'], '重建体素间距mm', 1e-6, 1e4)
        if 'lambda_smooth' in settings:
            clean['lambda_smooth'] = rc.number(settings['lambda_smooth'], '平滑权重', 0, 1e6)
        if 'iterations' in settings:
            clean['iterations'] = rc.integer(settings['iterations'], '迭代次数', 1, 60)
        if 'fbp_window' in settings:
            if settings['fbp_window'] not in ('hann', 'none'):
                raise InputError('FBP窗只支持hann或none')
            clean['fbp_window'] = settings['fbp_window']
        return clean

    def require_curating(session):
        if session['state'] != 'curating':
            raise Conflict('会话已' + ('确认' if session['state'] == 'confirmed' else '重建') + '，设置被冻结；请新建或克隆会话再试')

    @app.errorhandler(InputError)
    def recon_invalid(error):
        return jsonify({'error': str(error)}), 409 if isinstance(error, Conflict) else 400

    @app.get('/recon')
    def recon_home():
        return render_template('recon.html')

    @app.post('/core/projects', endpoint='recon_create_project')
    def recon_create_project():
        doc = store.create((request.get_json(silent=True) or {}).get('name', '投影重建检查'))
        return jsonify(doc)

    @app.get('/core/projects/<pid>', endpoint='recon_get_project')
    def recon_get_project(pid):
        return jsonify(store.load(pid))

    @app.get('/core/projects/<pid>/download', endpoint='recon_project_download')
    def recon_project_download(pid):
        return send_file(io.BytesIO(store.export_project(store.load(pid))),
                         mimetype='application/zip', as_attachment=True, download_name='corescan-project.zip')

    @app.post('/core/projects/<pid>/recon/sessions')
    def create_session(pid):
        bundle_file = request.files.get('bundle')
        angle_file = request.files.get('angles')
        if bundle_file is None or angle_file is None:
            raise InputError('需要投影ZIP（含projections/dark/flat）和角度表CSV')
        if len(bundle_file.read()) > 96 * 1024 * 1024:
            raise InputError('上传文件不能超过96MB')
        bundle_file.stream.seek(0)
        bundle = rc.load_bundle(bundle_file.read(), angle_file.read(), angle_file.filename)
        settings = validate_settings(json.loads(request.form.get('settings') or '{}'), bundle['shape'])
        defaults = {'direction': 1, 'transmission_floor': 1e-4, 'truncate_border': 0,
                    'grid_n': bundle['shape'][1], 'lambda_smooth': 0.05, 'iterations': 15, 'fbp_window': 'hann'}
        defaults.update(settings)
        missing = [key for key in ('du_mm', 'dz_mm', 'center') if key not in defaults]
        if missing:
            raise InputError('缺少必需几何参数：' + '、'.join(missing))
        revision = int(request.form['revision'])

        def add(current):
            current.setdefault('recon_sessions', [])
            session = {'id': uuid.uuid4().hex, 'name': str(request.form.get('name', '投影重建会话'))[:120],
                       'created': time.time(), 'state': 'curating', 'view_names': bundle['names'],
                       'angles': bundle['angles'], 'excluded': [False] * len(bundle['names']),
                       'holdout': [], 'settings': defaults, 'angle_warnings': bundle['unmatched_table_rows'],
                       'arrays': {'inputs': {'raw': put_arr(bundle['raw']), 'dark': put_arr(bundle['dark']),
                                             'flat': put_arr(bundle['flat'])}, 'corrected': {}, 'support': {},
                                  'schemes': {}, 'comparison': {}, 'roundtrip': {}},
                       'evidence': None, 'confirmation': None, 'roundtrips': []}
            line, mask, correction = rc.correct_fields(bundle['raw'], bundle['dark'], bundle['flat'],
                                                       defaults.get('saturation'), defaults['transmission_floor'],
                                                       defaults['truncate_border'])
            session['arrays']['corrected'] = {'line': put_arr(line), 'mask': put_arr(mask)}
            session['correction'] = correction
            current['recon_sessions'].append(session)
            return public(session)
        doc, result = store.mutate(pid, revision, add)
        return jsonify({'project': doc, 'result': result})

    @app.get('/core/projects/<pid>/recon/sessions')
    def list_sessions(pid):
        doc = store.load(pid)
        return jsonify([public(s) for s in doc.get('recon_sessions', [])])

    @app.get('/core/projects/<pid>/recon/sessions/<sid>')
    def get_session(pid, sid):
        return jsonify(public(find_session(store.load(pid), sid)))

    @app.post('/core/projects/<pid>/recon/sessions/<sid>/curation')
    def curate(pid, sid):
        p = body()

        def change(current):
            session = find_session(current, sid)
            require_curating(session)
            n_views = len(session['view_names'])
            if 'excluded' in p:
                excluded = p['excluded']
                if not isinstance(excluded, list) or len(excluded) != n_views:
                    raise InputError('剔除集合长度须等于视图数')
                session['excluded'] = [bool(v) for v in excluded]
            if 'holdout' in p:
                holdout = sorted(set(int(v) for v in p['holdout']))
                if any(v < 0 or v >= n_views for v in holdout):
                    raise InputError('留出视图索引越界')
                if any(session['excluded'][v] for v in holdout):
                    raise InputError('已剔除的视图不能同时作为独立留出')
                session['holdout'] = holdout
            if 'angles' in p:
                angles = p['angles']
                if not isinstance(angles, list) or len(angles) != n_views:
                    raise InputError('角度覆盖长度须等于视图数')
                session['angles'] = [None if a is None else rc.number(a, '角度(度)', -1e6, 1e6) for a in angles]
            updates = validate_settings(p.get('settings', {}), None)
            session['settings'].update(updates)
            raw = get_arr(session, 'inputs', 'raw')
            dark = get_arr(session, 'inputs', 'dark')
            flat = get_arr(session, 'inputs', 'flat')
            line, mask, correction = rc.correct_fields(raw, dark, flat, session['settings'].get('saturation'),
                                                       session['settings']['transmission_floor'],
                                                       session['settings']['truncate_border'])
            session['arrays']['corrected'] = {'line': put_arr(line), 'mask': put_arr(mask)}
            session['correction'] = correction
            return public(session)
        doc, result = store.mutate(pid, p.get('revision'), change)
        return jsonify({'project': doc, 'result': result})

    @app.post('/core/projects/<pid>/recon/sessions/<sid>/reconstruct')
    def reconstruct(pid, sid):
        p = body()

        def change(current):
            session = find_session(current, sid)
            require_curating(session)
            if any(a is None for a in session['angles']):
                raise InputError('仍有投影缺少角度配对，不能重建')
            if len([e for e in session['excluded'] if not e]) < 3:
                raise InputError('参与重建的有效视图少于3张，无法恢复切片')
            raw = get_arr(session, 'inputs', 'raw')
            dark = get_arr(session, 'inputs', 'dark')
            flat = get_arr(session, 'inputs', 'flat')
            line = get_arr(session, 'corrected', 'line')
            mask = get_arr(session, 'corrected', 'mask').astype(bool)
            outcome = rc.reconstruct_pipeline(raw, dark, flat, session['angles'], session['excluded'],
                                              {**session['settings'], 'holdout': session['holdout']})
            session['arrays']['corrected'] = {'line': put_arr(line), 'mask': put_arr(mask),
                                              'training_mask': put_arr(outcome['training_mask'])}
            session['arrays']['support'] = {'status': put_arr(outcome['support']['status']),
                                            'ray_fraction': put_arr(outcome['support']['ray_fraction'])}
            session['arrays']['schemes'] = {}
            for name in ('direct', 'constrained'):
                res = outcome['residuals'][name]
                session['arrays']['schemes'].update({
                    name + '_volume': put_arr(outcome['volumes'][name]),
                    name + '_forward': put_arr(res['forward']),
                    name + '_residual_train': put_arr(res['train']),
                    name + '_residual_all': put_arr(res['all_valid'])})
            session['arrays']['comparison'] = {'difference': put_arr(outcome['difference'])}
            angles = np.mod(np.asarray(session['angles'], dtype=float), 360.0)
            duplicate_holdout = []
            for v in session['holdout']:
                if any(i != v and not session['excluded'][i] and np.isclose(angles[i], angles[v], atol=1e-6)
                       for i in range(len(angles))):
                    duplicate_holdout.append(v)
            session['evidence'] = {'metrics': outcome['metrics'],
                                   'support_summary': {k: v for k, v in outcome['support'].items()
                                                       if k not in ('status', 'ray_fraction')},
                                   'fbp_scale': outcome['fbp_scale'], 'fit_info': outcome['fit_info'],
                                   'duplicate_holdout_views': duplicate_holdout,
                                   'interpretation': '更低的拟合误差或更平滑不自动代表更可信；请同时核对留出投影一致性、小孔薄壁条纹、边缘损失与支持范围。'}
            session['geometry'] = outcome['geometry']
            session['state'] = 'reconstructed'
            return public(session)
        doc, result = store.mutate(pid, p.get('revision'), change)
        return jsonify({'project': doc, 'result': result})

    @app.post('/core/projects/<pid>/recon/sessions/<sid>/confirm')
    def confirm(pid, sid):
        p = body()
        scheme = p.get('scheme')
        if scheme not in ('direct', 'constrained'):
            raise InputError('请选择 direct（直接恢复）或 constrained（约束拟合）方案')

        def change(current):
            session = find_session(current, sid)
            if session['state'] == 'curating':
                raise Conflict('请先运行双方案重建并核对证据')
            if session['state'] == 'confirmed':
                raise Conflict('会话已确认，确认体不可改写；需要新结果请新建或克隆会话')
            volume = get_arr(session, 'schemes', scheme + '_volume')
            status = get_arr(session, 'support', 'status')
            valid = (status == 2)
            geometry = session['geometry']
            record = store.add_volume(current, volume, '重建确认体 · ' + scheme + ' · ' + session['name'],
                                      {'spacing': [geometry['dz_mm'], geometry['dx_mm'], geometry['dx_mm']],
                                       'origin': geometry['origin_mm'], 'unit': 'attenuation',
                                       'description': '由平行束投影重建；可靠支持外的体素标记为无效，未决区域不是实测证实结构。'},
                                      valid, [])
            record['recon_source'] = {'session_id': session['id'], 'scheme': scheme, 'frozen_at': time.time()}
            session['confirmation'] = {'scheme': scheme, 'volume_id': record['id'], 'at': time.time(),
                                       'reason': str(p.get('reason', ''))[:2000],
                                       'evidence_fingerprint': json.dumps(session['evidence']['metrics'], sort_keys=True)}
            session['state'] = 'confirmed'
            return {'session': public(session), 'volume_id': record['id']}
        doc, result = store.mutate(pid, p.get('revision'), change)
        return jsonify({'project': doc, 'result': result})

    @app.post('/core/projects/<pid>/recon/sessions/<sid>/reproject')
    def reproject(pid, sid):
        p = body()

        def change(current):
            session = find_session(current, sid)
            if session['state'] == 'curating':
                raise Conflict('请先完成重建')
            scheme = p.get('scheme') or (session['confirmation']['scheme'] if session['confirmation'] else 'direct')
            if scheme not in ('direct', 'constrained'):
                raise InputError('方案只支持direct或constrained')
            volume = get_arr(session, 'schemes', scheme + '_volume')
            line = get_arr(session, 'corrected', 'line')
            mask = get_arr(session, 'corrected', 'mask').astype(bool)
            trip = rc.reproject(volume, session['angles'], session['settings'], line, mask,
                                np.asarray(session['excluded'], bool), np.asarray(session['holdout'], int))
            session['arrays']['roundtrip'] = {'forward': put_arr(trip['forward']),
                                              'residual': put_arr(trip['residual'])}
            entry = {'at': time.time(), 'scheme': scheme, 'metrics': trip['metrics'],
                     'per_view': trip['per_view'], 'note': trip['ordering_note']}
            session.setdefault('roundtrips', []).append(entry)
            return {'session': public(session), 'trip': entry}
        doc, result = store.mutate(pid, p.get('revision'), change)
        return jsonify({'project': doc, 'result': result})

    @app.post('/core/projects/<pid>/recon/sessions/<sid>/clone')
    def clone_session(pid, sid):
        p = body()

        def change(current):
            source = find_session(current, sid)
            clone = json.loads(json.dumps(source, ensure_ascii=False))
            clone['id'] = uuid.uuid4().hex
            clone['name'] = source['name'] + ' · 新设置副本'
            clone['created'] = time.time()
            clone['state'] = 'curating'
            clone['evidence'] = None
            clone['confirmation'] = None
            clone['roundtrips'] = []
            clone['arrays'] = {'inputs': dict(source['arrays']['inputs']), 'corrected': {}, 'support': {},
                               'schemes': {}, 'comparison': {}, 'roundtrip': {}}
            current.setdefault('recon_sessions', []).append(clone)
            raw = get_arr(clone, 'inputs', 'raw')
            dark = get_arr(clone, 'inputs', 'dark')
            flat = get_arr(clone, 'inputs', 'flat')
            line, mask, correction = rc.correct_fields(raw, dark, flat, clone['settings'].get('saturation'),
                                                       clone['settings']['transmission_floor'],
                                                       clone['settings']['truncate_border'])
            clone['arrays']['corrected'] = {'line': put_arr(line), 'mask': put_arr(mask)}
            clone['correction'] = correction
            return public(clone)
        doc, result = store.mutate(pid, p.get('revision'), change)
        return jsonify({'project': doc, 'result': result})

    @app.get('/core/projects/<pid>/recon/sessions/<sid>/image/<group>/<name>')
    def session_image(pid, sid, group, name):
        session = find_session(store.load(pid), sid)
        if group not in session['arrays'] or name not in session['arrays'][group]:
            raise InputError('数组不存在')
        arr = store.array(session['arrays'][group][name])
        projection_groups = ('inputs', 'corrected', 'roundtrip')
        is_status = group == 'support' and name == 'status'
        if arr.ndim == 3 and request.args.get('view') is not None:
            view = int(request.args['view'])
            arr = arr[min(max(view, 0), arr.shape[0] - 1)]
        elif arr.ndim == 3 and group in projection_groups and name != 'training_mask':
            arr = arr[min(int(request.args.get('view', 0)), arr.shape[0] - 1)]
        elif arr.ndim == 3:
            axis = {'z': 0, 'y': 1, 'x': 2}.get(request.args.get('axis', 'z'), 0)
            index = int(request.args.get('index', arr.shape[axis] // 2))
            arr = np.take(arr, min(max(index, 0), arr.shape[axis] - 1), axis=axis)
        if is_status:
            palette = np.array([[12, 18, 25], [196, 142, 52], [74, 190, 158]], dtype=np.uint8)
            rgb = palette[np.clip(arr.astype(int), 0, 2)]
            output = io.BytesIO()
            Image.fromarray(rgb).save(output, format='PNG')
            raw_png = output.getvalue()
        else:
            raw_png = png_bytes(arr.astype(np.float32))
        return Response(raw_png, mimetype='image/png', headers={'Cache-Control': 'no-store'})

    @app.get('/core/projects/<pid>/recon/sessions/<sid>/download')
    def session_download(pid, sid):
        doc = store.load(pid)
        session = find_session(doc, sid)
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('session.json', json.dumps(public(session), ensure_ascii=False, indent=2))
            for group, names in session['arrays'].items():
                for name, key in names.items():
                    archive.writestr(group + '/' + name + '.npy', npy_bytes(store.array(key)))
            if session.get('evidence'):
                for scheme, metrics in session['evidence']['metrics'].items():
                    for table_name in ('per_view',):
                        text = io.StringIO()
                        writer = csv.DictWriter(text, fieldnames=['view', 'role', 'rmse', 'mae'])
                        writer.writeheader()
                        writer.writerows(metrics[table_name])
                        archive.writestr(scheme + '_' + table_name + '.csv', text.getvalue().encode('utf-8-sig'))
        return send_file(io.BytesIO(output.getvalue()), mimetype='application/zip', as_attachment=True,
                         download_name='recon-session.zip')
