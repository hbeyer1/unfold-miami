"""Serve this folder on localhost (the page, src/ and the baked data/).

  python3 serve.py [--port 8780]    then open http://localhost:8780
"""
import argparse
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
mimetypes.add_type("application/octet-stream", ".bin")
mimetypes.add_type("text/javascript", ".js")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    port = ap.parse_args().port
    print(f"Unfold · Miami Beach on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
