import os
from image_lab import create_app
from corescan.server import install
from waitress import serve
root = os.environ.get('CORESCAN_DATA_DIR', 'data')
app = create_app(os.path.join(root, 'preparation'))
install(app, os.path.join(root, 'volumes'))
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', '18925')), threads=4)
