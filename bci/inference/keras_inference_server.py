import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import numpy as np
import os

def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def _set_json(self, code=200):
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

        def do_GET(self):
            if self.path == '/health':
                self._set_json(200)
                self.wfile.write(json.dumps({'status': 'ok'}).encode('utf-8'))
            else:
                self._set_json(404)
                self.wfile.write(json.dumps({'error': 'not_found'}).encode('utf-8'))

        def do_POST(self):
            if self.path == '/predict':
                length = int(self.headers.get('Content-Length', 0))
                data = self.rfile.read(length)
                try:
                    j = json.loads(data.decode('utf-8'))
                    X = np.array(j.get('data'))
                    # Ensure shape (samples, time_steps, channels)
                    if X.ndim == 2:
                        X = X.reshape(1, X.shape[0], X.shape[1])
                    probs = model.predict(X, verbose=0).tolist()
                    self._set_json(200)
                    self.wfile.write(json.dumps({'probs': probs}).encode('utf-8'))
                except Exception as e:
                    self._set_json(500)
                    self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            else:
                self._set_json(404)
                self.wfile.write(json.dumps({'error': 'not_found'}).encode('utf-8'))

    return Handler

def run_server(model_path, host='127.0.0.1', port=5001):
    # import tensorflow here (subprocess will have clean interpreter)
    import tensorflow as tf
    from tensorflow import keras

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = keras.models.load_model(model_path)

    handler = make_handler(model)
    httpd = HTTPServer((host, port), handler)
    print(f'Keras inference server listening on {host}:{port} for model {model_path}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .keras model')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    run_server(args.model, port=args.port)
