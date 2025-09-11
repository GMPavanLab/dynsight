import functools
import logging
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer

logger = logging.getLogger(__name__)


class HTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass

    # do_POST must be uppercase
    def do_POST(self) -> None:
        if self.path == "/shutdown":
            self.send_response(200)
            self.end_headers()
            logger.info("Shutdown request received.")
            threading.Thread(target=self.server.shutdown).start()
        else:
            self.send_error(404)


class ReusableTCPServer(TCPServer):
    allow_reuse_address = True


def label_tool(port: int = 8888) -> None:
    web_dir = Path(__file__).parent / "label_tool"
    handler = functools.partial(HTTPRequestHandler, directory=str(web_dir))
    with ReusableTCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/index.html"
        logger.info(f"Starting server at {url}")
        webbrowser.open(url)
        httpd.serve_forever()
        httpd.server_close()
        logger.info("Server closed.")
