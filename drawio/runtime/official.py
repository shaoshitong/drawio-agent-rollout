"""Run the pinned official editor offline with task-local browser resources."""

from contextlib import ExitStack, contextmanager
import http.server
import mimetypes
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import urllib.parse
import zipfile

from playwright.sync_api import sync_playwright

RESOURCES = Path(os.environ.get("DRAWIO_RUNTIME", ".runtime"))
WAR = RESOURCES / "draw.war"
CHROME = str(RESOURCES / "chrome-linux64/chrome")
MATH_BOOTSTRAP = r"""async origin => {
    window.DRAW_MATH_URL = origin + '/math4/es5';
    if (typeof Editor.loadMath !== 'function')
        Editor.initMath(origin + '/math4/es5/drawio-mathjax.min.js');
    if (typeof window.DrawioMathJax?.typeset !== 'function')
        window.DrawioMathJax.svg = {fontCache: 'none'};
    await new Promise(resolve => Editor.loadMath(resolve));
    if (typeof window.DrawioMathJax?.typeset !== 'function')
        throw Error('official_local_MathJax_not_ready');
    window.DrawioMathJax.startup.output.options.fontCache = 'none';
}"""


@contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class AssetHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        owner = self.server.owner
        key = urllib.parse.unquote(urllib.parse.urlparse(self.path).path).lstrip("/") or "index.html"
        try:
            if key in owner.assets:
                body = owner.assets[key].read_bytes()
            else:
                with owner.lock:
                    body = owner.archive.read(key)
            if key == "js/main.js":
                body = b"App.main(function(ui){window.__bulkUi=ui;});\n" + body
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(key)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (KeyError, FileNotFoundError):
            owner.missing.append(key)
            self.send_error(404)

    def log_message(self, *args):
        pass


class OfflineEditor:
    def __init__(self):
        self.stack = ExitStack()
        self.lock = threading.Lock()
        self.assets, self.blocked, self.missing = {}, [], []
        try:
            home = self.stack.enter_context(
                TemporaryDirectory(prefix="dio-", dir=os.environ.get("DSH_CHROME_TMPDIR", "/tmp"))
            )
            self.archive = self.stack.enter_context(zipfile.ZipFile(WAR))
            self.server = self.stack.enter_context(
                http.server.ThreadingHTTPServer(("127.0.0.1", 0), AssetHandler)
            )
            self.server.owner = self
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            self.stack.callback(self.server.shutdown)
            self.origin = "http://127.0.0.1:" + str(self.server.server_port)
            values = {key: home for key in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "TMPDIR")}
            values.update(
                FONTCONFIG_FILE=str(WAR.parent / "fonts.conf"),
                LD_LIBRARY_PATH=str(WAR.parent / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
            )
            with environment(values):
                self.playwright = sync_playwright().start()
                self.stack.callback(self.playwright.stop)
                self.browser = self.playwright.chromium.launch(
                    executable_path=CHROME, args=["--disable-dev-shm-usage", "--disable-gpu"]
                )
                self.stack.callback(self.browser.close)
                self.context = self.browser.new_context(service_workers="block")
                self.context.route("**/*", self.guard)
                self.page = self.context.new_page()
                self.page.goto(
                    self.origin + "/?dev=1&offline=1&local=1&splash=0&libs=0&math=1&ui=kennedy&stealth=1",
                    wait_until="domcontentloaded",
                    timeout=45000,
                )
                self.page.wait_for_function("window.__bulkUi && window.__bulkUi.editor", timeout=45000)
                self.page.evaluate(MATH_BOOTSTRAP, self.origin)
                self.render_page = self.context.new_page()
        except BaseException:
            self.stack.close()
            raise

    def guard(self, route):
        if route.request.url.startswith(self.origin + "/"):
            route.continue_()
        else:
            self.blocked.append(route.request.url.split("?", 1)[0])
            route.abort()

    def close(self):
        self.stack.close()
