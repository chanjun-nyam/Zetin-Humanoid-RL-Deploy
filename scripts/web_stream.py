import os

# offscreen rendering backend (must be set before mujoco creates a gl context).
# 'egl' for headless gpu servers, 'osmesa' for cpu-only. override via env var.
os.environ.setdefault('MUJOCO_GL', 'egl')

import io
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import mujoco as mj


# jpeg encoder: prefer opencv (fast), fall back to pillow
try:
    import cv2

    def encode_jpeg(rgb, quality):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes()

except ImportError:
    from PIL import Image

    def encode_jpeg(rgb, quality):
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format='JPEG', quality=int(quality))
        return buf.getvalue()



@dataclass
class WebStreamCfg:

    host: str = '0.0.0.0'

    port: int = 8000

    width: int = 640

    height: int = 480

    camera: str = ''

    jpeg_quality: int = 80

    stream_freq: int = 50



INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>sim-stream</title>
<style>
  body { margin: 0; padding: 16px; background: #111; color: #eee; font-family: monospace; }
  .wrap { display: flex; flex-direction: column; align-items: center; gap: 16px; }
  img { background: #000; border: 1px solid #333; max-width: 100%; }
  .sliders { width: __WIDTH__px; max-width: 100%; display: flex; flex-direction: column; gap: 12px; }
  .row { display: flex; align-items: center; gap: 12px; }
  .row label { width: 32px; }
  .row input { flex: 1; }
  .row span { width: 56px; text-align: right; }
</style>
</head>
<body>
  <div class="wrap">
    <img id="view" src="/stream" width="__WIDTH__" height="__HEIGHT__">
    <div class="sliders">
      <div class="row"><label>vx</label><input id="s0" type="range" min="-1" max="1" step="0.01" value="0"><span id="v0">0.00</span></div>
      <div class="row"><label>vy</label><input id="s1" type="range" min="-1" max="1" step="0.01" value="0"><span id="v1">0.00</span></div>
      <div class="row"><label>wz</label><input id="s2" type="range" min="-1" max="1" step="0.01" value="0"><span id="v2">0.00</span></div>
    </div>
  </div>
<script>
  const s = [0, 1, 2].map(i => document.getElementById('s' + i));
  const v = [0, 1, 2].map(i => document.getElementById('v' + i));
  let dirty = true;
  s.forEach((el, i) => el.addEventListener('input', () => {
    v[i].textContent = parseFloat(el.value).toFixed(2);
    dirty = true;
  }));
  setInterval(() => {
    if (!dirty) return;
    dirty = false;
    fetch(`/cmd?x=${s[0].value}&y=${s[1].value}&z=${s[2].value}`).catch(() => {});
  }, 50);
</script>
</body>
</html>
"""



class _Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass  # silence default request logging

    def do_GET(self):
        app = self.server.app
        parsed = urlparse(self.path)

        if parsed.path == '/':
            self._serve_index(app)
        elif parsed.path == '/stream':
            self._serve_stream(app)
        elif parsed.path == '/cmd':
            self._serve_cmd(app, parse_qs(parsed.query))
        else:
            self.send_error(404)

    def _serve_index(self, app):
        body = (
            INDEX_HTML
            .replace('__WIDTH__', str(app.cfg.width))
            .replace('__HEIGHT__', str(app.cfg.height))
            .encode()
        )
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self, app):
        self.send_response(200)
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        period = 1.0 / app.cfg.stream_freq
        try:
            while True:
                t0 = time.perf_counter()
                frame = app.get_latest_jpeg()
                if frame is not None:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected

    def _serve_cmd(self, app, query):
        try:
            values = [float(query[k][0]) for k in ('x', 'y', 'z')]
        except (KeyError, IndexError, ValueError):
            self.send_error(400)
            return
        app.set_command(values)
        self.send_response(204)
        self.end_headers()



class _HTTPServer(ThreadingHTTPServer):

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, app):
        super().__init__(address, _Handler)
        self.app = app



class WebStreamServer:

    def __init__(self, cfg: WebStreamCfg):
        self.cfg = cfg

        # camera: '' -> free camera (-1), otherwise a camera name from the mjcf
        self._camera = cfg.camera if cfg.camera else -1

        # lazily created in the sim/render thread (gl context is thread-local)
        self._renderer = None

        self._jpeg = None
        self._jpeg_lock = threading.Lock()

        self._cmd = [0.0, 0.0, 0.0]
        self._cmd_lock = threading.Lock()

        # http server in a background thread
        self._httpd = _HTTPServer((cfg.host, cfg.port), self)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f'[web-stream] serving on http://{cfg.host}:{cfg.port}')


    # registered on the simulator via register_render_callback
    def render_frame(self, mj_model, mj_data):
        if self._renderer is None:
            self._renderer = mj.Renderer(mj_model, height=self.cfg.height, width=self.cfg.width)
        self._renderer.update_scene(mj_data, camera=self._camera)
        jpeg = encode_jpeg(self._renderer.render(), self.cfg.jpeg_quality)
        with self._jpeg_lock:
            self._jpeg = jpeg


    # registered on the simulator via register_cmd_callback
    def get_cmd(self):
        with self._cmd_lock:
            return list(self._cmd)


    # called from the http handler thread
    def get_latest_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg


    # called from the http handler thread
    def set_command(self, values):
        clipped = [max(-1.0, min(1.0, float(v))) for v in values]
        with self._cmd_lock:
            self._cmd = clipped
