import io
import json
from copy import deepcopy
import numpy as np
from flask import request, jsonify, render_template, send_file, Response
from .core import InputError, decode, metadata, png_bytes, npy_bytes, axis_index, integer
from .storage import Store, Conflict
from . import volume_ops as vo
from . import segmentation as seg
from .demos import make_demo

OPERATIONS = {
    'projection': {'name': '正交投影与厚层平均', 'description': '沿一个体素轴汇总；显示支持计数。积分不补造无效位置。', 'minimum': 1, 'parameters': {'axis': 'z', 'method': 'maximum'}},
    'crop': {'name': '物理坐标裁剪', 'description': '裁剪局部体数据，并把原点移动到正确的体素中心。范围右端不含。', 'minimum': 1, 'parameters': {'bounds': [4, 44, 10, 70, 10, 78]}},
    'gaussian': {'name': '物理半径平滑', 'description': '使用毫米半径和有效支持归一化；原始体数据保留。', 'minimum': 1, 'parameters': {'sigma_mm': .4, 'minimum_support': .75}},
    'gradient': {'name': '三维边缘强度', 'description': '按各轴体素间距计算梯度；邻域不完整的位置无效。', 'minimum': 1, 'parameters': {}},
    'histogram': {'name': '体素分布统计', 'description': '仅统计有效体素的强度分布和分位数。', 'minimum': 1, 'parameters': {'bins': 64}},
    'profiles': {'name': '逐层统计曲线', 'description': '生成Z/Y/X逐层统计，保留物理位置、有效计数及空层。', 'minimum': 1, 'parameters': {}},
    'resample': {'name': '规则体素重采样', 'description': '按新体素间距重新采样，保持首体素原点。标签请用nearest。', 'minimum': 1, 'parameters': {'spacing': [.4, .4, .4], 'mode': 'linear'}},
    'compare': {'name': '同网格数值比较', 'description': '对形状、间距、原点及单位一致的两体做差；不自动对齐。', 'minimum': 2, 'parameters': {}},
    'line': {'name': '三维直线剖面', 'description': '两端使用Z/Y/X体素索引；横轴显示真实毫米距离。', 'minimum': 1, 'parameters': {'start': [24, 40, 4], 'end': [24, 40, 83], 'samples': 128}},
    'threshold': {'name': '强度阈值分区', 'description': '在有效支持内选择指定强度范围，得到独立二值体。', 'minimum': 1, 'parameters': {'lower': .6, 'upper': 2, 'invert': False}},
    'morphology': {'name': '二值体形态整理', 'description': '按物理半径开闭、膨胀、腐蚀，或填充内部空腔。', 'minimum': 1, 'binary': True, 'parameters': {'operation': 'closing', 'radius_mm': .5}},
    'components': {'name': '三维连通对象测量', 'description': '输出稳定排序标签、物理体积、中心、边界状态与体素面面积。', 'minimum': 1, 'binary': True, 'parameters': {'connectivity': 6, 'minimum_volume': .1}},
    'select': {'name': '提取已标号对象', 'description': '把所选标签单独生成二值体，便于查看或继续测量。', 'minimum': 1, 'labels': True, 'parameters': {'labels': [1]}},
    'exclude': {'name': '排除不可靠体区域', 'description': '改变有效支持并保留原始数值，后续运算不会把排除区当实测。', 'minimum': 1, 'parameters': {'bounds': [0, 4, 0, 80, 0, 88]}},
    'region': {'name': '局部体积统计', 'description': '统计一个局部长方体的有效体积和强度分布，不改变原体。', 'minimum': 1, 'parameters': {'bounds': [8, 40, 15, 65, 15, 73]}}
}


def execute(store, doc, operation, ids, parameters):
    if operation not in OPERATIONS:
        raise InputError('未知操作')
    spec = OPERATIONS[operation]
    if not isinstance(ids, list) or len(ids) != spec['minimum'] or len(set(ids)) != len(ids):
        raise InputError('此操作需要恰好' + str(spec['minimum']) + '份不同体数据')
    if not isinstance(parameters, dict):
        raise InputError('参数需要JSON对象')
    records = [store.find(doc, vid) for vid in ids]
    data, valid = store.arrays(records[0])
    meta = records[0]['meta']
    if spec.get('binary') and meta['unit'] != 'binary':
        raise InputError('请先选择二值分区结果')
    if spec.get('labels') and meta['unit'] != 'labels':
        raise InputError('请先选择连通对象标签结果')
    p = deepcopy(parameters)
    if operation == 'projection':
        result = vo.projection(data, valid, meta, p.get('axis', 'z'), p.get('method', 'maximum'), p.get('start'), p.get('end'))
    elif operation == 'crop':
        result = vo.crop(data, valid, meta, p.get('bounds'))
    elif operation == 'gaussian':
        result = vo.gaussian(data, valid, meta, p.get('sigma_mm', .4), p.get('minimum_support', .75))
    elif operation == 'gradient':
        result = vo.gradient(data, valid, meta)
    elif operation == 'histogram':
        result = vo.histogram(data, valid, p.get('bins', 64))
    elif operation == 'profiles':
        result = vo.profiles(data, valid, meta)
    elif operation == 'resample':
        mode = p.get('mode', 'linear')
        if meta['unit'] in ('labels', 'binary') and mode != 'nearest':
            raise InputError('标签或二值体需要最近邻重采样')
        result = vo.resample(data, valid, meta, p.get('spacing'), mode)
    elif operation == 'compare':
        second, second_valid = store.arrays(records[1])
        result = vo.compare(data, valid, meta, second, second_valid, records[1]['meta'])
    elif operation == 'line':
        result = vo.sample_line(data, valid, meta, p.get('start'), p.get('end'), p.get('samples', 128))
    elif operation == 'threshold':
        result = seg.threshold(data, valid, p.get('lower'), p.get('upper'), bool(p.get('invert', False)))
    elif operation == 'morphology':
        result = seg.morphology(data, valid, meta, p.get('operation', 'closing'), p.get('radius_mm', .5))
    elif operation == 'components':
        result = seg.components(data, valid, meta, p.get('connectivity', 6), p.get('minimum_volume', 0))
    elif operation == 'select':
        result = seg.select_labels(data, valid, p.get('labels'))
    elif operation == 'exclude':
        result = seg.exclude_box(data, valid, p.get('bounds'))
    else:
        result = seg.region_stats(data, valid, meta, p.get('bounds'))
    return store.save_run(doc, operation, ids, p, result)


def install(app, data_dir):
    store = Store(data_dir)
    app.extensions['core_store'] = store

    def body():
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            raise InputError('需要JSON对象')
        return value

    def mutate(pid, callback):
        p = body()
        doc, result = store.mutate(pid, p.get('revision'), lambda current: callback(current, p))
        return jsonify({'project': doc, 'result': result})

    @app.errorhandler(InputError)
    def invalid(error):
        return jsonify({'error': str(error)}), 409 if isinstance(error, Conflict) else 400

    @app.get('/')
    def core_home():
        return render_template('core.html')

    @app.get('/core/health')
    def core_health():
        return jsonify({'ok': True, 'product': 'CoreScan', 'data_dir': str(store.root), 'max_voxels': 8000000})

    @app.get('/core/operations')
    def operations():
        return jsonify(OPERATIONS)

    @app.get('/core/projects')
    def projects():
        return jsonify(store.listing())

    @app.post('/core/projects')
    def create():
        return jsonify(store.create(body().get('name', '工业检查')))

    @app.get('/core/projects/<pid>')
    def project(pid):
        return jsonify(store.load(pid))

    @app.post('/core/demo')
    def demo():
        doc = store.create('合成铸件 · 体数据检查')
        inputs, description = make_demo()
        def populate(current):
            for item in inputs:
                store.add_volume(current, item['data'], item['name'], item['meta'])
            current['demo'] = description
            current['notes'] = description['description']
        doc, _ = store.mutate(doc['id'], 0, populate)
        return jsonify(doc)

    @app.post('/core/projects/<pid>/upload')
    def upload(pid):
        file = request.files.get('file')
        if file is None:
            raise InputError('请选择体数据文件')
        data, meta, good = decode(file.read(), file.filename)
        raw_meta = request.form.get('metadata')
        if raw_meta:
            try:
                meta.update(json.loads(raw_meta))
            except ValueError as error:
                raise InputError('元信息JSON无效') from error
        revision = integer(request.form.get('revision'), '版本', 0, 100000000)
        doc, result = store.mutate(pid, revision, lambda current: store.add_volume(current, data, file.filename, meta, good))
        return jsonify({'project': doc, 'result': result})

    @app.post('/core/projects/<pid>/metadata/<vid>')
    def update_metadata(pid, vid):
        def change(doc, p):
            old = store.find(doc, vid)
            data, good = store.arrays(old)
            meta = dict(old['meta'])
            meta.update(p.get('metadata', {}))
            return store.add_volume(doc, data, p.get('name', old['name'] + ' · 元信息副本'), meta, good, [vid])
        return mutate(pid, change)

    @app.post('/core/projects/<pid>/run')
    def process(pid):
        return mutate(pid, lambda doc, p: execute(store, doc, p.get('operation'), p.get('source_ids'), p.get('parameters', {})))

    @app.get('/core/projects/<pid>/volumes/<vid>/slice')
    def slice_view(pid, vid):
        record = store.find(store.load(pid), vid)
        data, valid = store.arrays(record)
        axis = request.args.get('axis', 'z')
        view, good = vo.section(data, valid, axis, request.args.get('index'))
        raw = png_bytes(view, good, request.args.get('low'), request.args.get('high'), labels=record['meta']['unit'] == 'labels')
        return Response(raw, mimetype='image/png', headers={'Cache-Control': 'no-store'})

    @app.get('/core/projects/<pid>/volumes/<vid>/download')
    def volume_download(pid, vid):
        record = store.find(store.load(pid), vid)
        data, good = store.arrays(record)
        output = io.BytesIO()
        np.savez_compressed(output, volume=data, valid=good, spacing=record['meta']['spacing'], origin=record['meta']['origin'], metadata_json=json.dumps(record['meta']))
        return send_file(io.BytesIO(output.getvalue()), mimetype='application/octet-stream', as_attachment=True, download_name='volume.npz')

    @app.get('/core/projects/<pid>/runs/<rid>/image/<kind>/<name>')
    def run_image(pid, rid, kind, name):
        run = store.run(store.load(pid), rid)
        if kind not in ('images', 'maps') or name not in run[kind]:
            raise InputError('结果图不存在')
        arr = store.array(run[kind][name])
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        return Response(png_bytes(arr), mimetype='image/png')

    @app.get('/core/projects/<pid>/runs/<rid>/download')
    def result_download(pid, rid):
        raw = store.export_run(store.load(pid), rid)
        return send_file(io.BytesIO(raw), mimetype='application/zip', as_attachment=True, download_name='inspection-result.zip')

    @app.post('/core/projects/<pid>/snapshot')
    def snapshot(pid):
        return mutate(pid, lambda doc, p: store.snapshot(doc, p.get('name', '检查快照')))

    @app.post('/core/projects/<pid>/notes')
    def notes(pid):
        def update(doc, p):
            doc['notes'] = str(p.get('notes', ''))[:20000]
            return {'notes': doc['notes']}
        return mutate(pid, update)

    @app.get('/core/projects/<pid>/download')
    def project_download(pid):
        return send_file(io.BytesIO(store.export_project(store.load(pid))), mimetype='application/zip', as_attachment=True, download_name='corescan-project.zip')
