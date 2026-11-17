import csv
import io
import json
import time
import uuid
import hashlib
import threading
import zipfile
from pathlib import Path
from copy import deepcopy
import numpy as np
from .core import InputError, volume, validity, metadata, npy_bytes, png_bytes, summary_stats
from .volume_ops import section


class Conflict(InputError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.documents = self.root / 'projects'
        self.objects = self.root / 'objects'
        self.documents.mkdir(parents=True, exist_ok=True)
        self.objects.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def key(self, value):
        value = str(value)
        if not value or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in value):
            raise InputError('标识无效')
        return value

    def write(self, path, data):
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(path)

    def load(self, pid):
        path = self.documents / (self.key(pid) + '.json')
        if not path.exists():
            raise InputError('检查项目不存在')
        return json.loads(path.read_text())

    def create(self, name):
        with self.lock:
            now = time.time()
            doc = {'id': uuid.uuid4().hex, 'name': str(name or '工业切片检查')[:120], 'revision': 0, 'created': now, 'updated': now, 'volumes': [], 'runs': [], 'notes': '', 'snapshots': []}
            self.write(self.documents / (doc['id'] + '.json'), doc)
            return doc

    def listing(self):
        with self.lock:
            docs = [json.loads(p.read_text()) for p in self.documents.glob('*.json')]
        return sorted([{'id': d['id'], 'name': d['name'], 'revision': d['revision'], 'volumes': len(d['volumes']), 'runs': len(d['runs']), 'updated': d['updated']} for d in docs], key=lambda d: d['updated'], reverse=True)

    def mutate(self, pid, revision, callback):
        with self.lock:
            doc = self.load(pid)
            if revision != doc['revision']:
                raise Conflict('项目已变化，请刷新后重试')
            result = callback(doc)
            doc['revision'] += 1
            doc['updated'] = time.time()
            self.write(self.documents / (doc['id'] + '.json'), doc)
            return doc, result

    def put(self, data):
        raw = npy_bytes(data)
        key = hashlib.sha256(raw).hexdigest()
        path = self.objects / (key + '.npy')
        if not path.exists():
            temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
            temporary.write_bytes(raw)
            temporary.replace(path)
        return key

    def array(self, key):
        path = self.objects / (self.key(key) + '.npy')
        if not path.exists():
            raise InputError('体素对象不存在')
        return np.load(path, allow_pickle=False)

    def add_volume(self, doc, data, name, meta=None, valid=None, parents=None):
        data = volume(data)
        good = validity(valid, data.shape)
        record = {'id': uuid.uuid4().hex, 'name': str(name)[:160], 'asset': self.put(data), 'valid': self.put(good), 'shape': list(data.shape), 'meta': metadata(meta), 'parents': parents or [], 'created': time.time(), 'statistics': summary_stats(data, good)}
        doc['volumes'].append(record)
        return record

    def find(self, doc, vid):
        record = next((v for v in doc['volumes'] if v['id'] == vid), None)
        if record is None:
            raise InputError('体数据不存在')
        return record

    def arrays(self, record):
        return self.array(record['asset']), self.array(record['valid']).astype(bool)

    def run(self, doc, rid):
        record = next((r for r in doc['runs'] if r['id'] == rid), None)
        if record is None:
            raise InputError('处理结果不存在')
        return record

    def save_run(self, doc, operation, source_ids, parameters, result):
        sources = [deepcopy(self.find(doc, vid)) for vid in source_ids]
        record = {'id': uuid.uuid4().hex, 'operation': operation, 'created': time.time(), 'sources': sources, 'parameters': deepcopy(parameters), 'report': result.get('report', {}), 'tables': result.get('tables', {}), 'images': {}, 'maps': {}, 'derived_volume': None}
        for kind in ('images', 'maps'):
            for name, arr in result.get(kind, {}).items():
                arr = np.asarray(arr, np.float32)
                if not np.isfinite(arr).all():
                    raise InputError('结果含非有限数值')
                record[kind][name] = self.put(arr)
        if 'volume' in result:
            meta = deepcopy(result.get('meta', sources[0]['meta']))
            meta.update(result.get('meta_update', {}))
            derived = self.add_volume(doc, result['volume'], operation + ' · ' + sources[0]['name'], meta, result.get('valid'), source_ids)
            record['derived_volume'] = derived['id']
        doc['runs'].append(record)
        return record

    def snapshot(self, doc, label):
        snap = {'id': uuid.uuid4().hex, 'name': str(label)[:120], 'created': time.time(), 'revision': doc['revision'], 'volumes': deepcopy(doc['volumes']), 'run_ids': [r['id'] for r in doc['runs']], 'notes': doc['notes']}
        doc['snapshots'].append(snap)
        return snap

    def export_run(self, doc, rid):
        run = self.run(doc, rid)
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('report.json', json.dumps(run, ensure_ascii=False, indent=2))
            for name, rows in run['tables'].items():
                if not rows:
                    continue
                text = io.StringIO()
                keys = list(dict.fromkeys(k for row in rows for k in row))
                writer = csv.DictWriter(text, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
                archive.writestr(name + '.csv', text.getvalue().encode('utf-8-sig'))
            for category in ('maps', 'images'):
                for name, key in run[category].items():
                    arr = self.array(key)
                    archive.writestr(category + '/' + name + '.npy', npy_bytes(arr))
                    view = arr[arr.shape[0] // 2] if arr.ndim == 3 else arr
                    archive.writestr(category + '/' + name + '.png', png_bytes(view))
            if run['derived_volume']:
                record = self.find(doc, run['derived_volume'])
                data, valid = self.arrays(record)
                archive.writestr('volume.npy', npy_bytes(data))
                archive.writestr('valid.npy', npy_bytes(valid))
                archive.writestr('metadata.json', json.dumps(record['meta'], ensure_ascii=False, indent=2))
                for axis in ('z', 'y', 'x'):
                    view, mask = section(data, valid, axis)
                    archive.writestr('section_' + axis + '.png', png_bytes(view, mask, labels=record['meta']['unit'] == 'labels'))
        return output.getvalue()

    def export_project(self, doc):
        output = io.BytesIO()
        keys = set()
        for v in doc['volumes']:
            keys.update([v['asset'], v['valid']])
        for run in doc['runs']:
            keys.update(run['images'].values())
            keys.update(run['maps'].values())
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('project.json', json.dumps(doc, ensure_ascii=False, indent=2))
            for key in sorted(keys):
                archive.writestr('objects/' + key + '.npy', npy_bytes(self.array(key)))
        return output.getvalue()
