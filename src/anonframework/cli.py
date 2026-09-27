"""``anonframework`` command-line interface.

One command today: ``anonframework serve`` starts a local web server over a results
directory and opens the dashboard in your browser, so you never run
``python -m http.server`` by hand::

    anonframework serve ./anonframework-results

The results website is a static page that fetches the JSON in its directory;
browsers block those fetches from ``file://``, hence a tiny local server.  The
freshest bundled ``dashboard.html`` is (re)written into the directory on serve,
so it always matches the installed version and works on older result trees too.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import importlib.resources
import sys
import threading
import webbrowser
from pathlib import Path

DEFAULT_RESULTS_DIR = "anonframework-results"


def _bundled_dashboard() -> bytes | None:
    try:
        return (importlib.resources.files("anonframework.core") / "dashboard.html").read_bytes()
    except Exception:  # pragma: no cover - the asset always ships in the wheel
        return None


def ensure_dashboard(root: Path) -> bool:
    """Write the bundled ``dashboard.html`` into *root*; return whether it landed."""
    data = _bundled_dashboard()
    if data is None:  # pragma: no cover - asset always present
        return False
    try:
        (root / "dashboard.html").write_bytes(data)
        return True
    except OSError:  # pragma: no cover - unwritable target
        return False


def looks_like_results(root: Path) -> bool:
    """Whether *root* has the markers of a anonframework results dir (root or experiment)."""
    return any((root / name).exists() for name in ("experiments.json", "manifest.json"))


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return  # silence the per-request stderr chatter


def make_server(host: str, port: int, directory: Path) -> http.server.ThreadingHTTPServer:
    """Bind an HTTP server on *port*, falling back to a free port if it's taken."""
    handler = functools.partial(_QuietHandler, directory=str(directory))
    try:
        return http.server.ThreadingHTTPServer((host, port), handler)
    except OSError:
        return http.server.ThreadingHTTPServer((host, 0), handler)


def serve(
    directory: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> int:
    """Serve *directory* over HTTP and open the dashboard. Blocks until Ctrl-C."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        print(f"anonframework: no such results directory: {root}", file=sys.stderr)
        return 2
    ensure_dashboard(root)
    if not looks_like_results(root):
        print(
            f"anonframework: {root} has no experiments.json / manifest.json; "
            "serving it anyway (is this the right results directory?)",
            file=sys.stderr,
        )
    httpd = make_server(host, port, root)
    url = f"http://{host}:{httpd.server_address[1]}/dashboard.html"
    print(f"anonframework: serving {root}\n  {url}\n  (press Ctrl-C to stop)", flush=True)
    if open_browser:  # pragma: no cover - opens a real browser
        threading.Timer(0.4, lambda: webbrowser.open(url, new=2)).start()
    try:  # pragma: no cover - blocking serve loop
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nanonframework: stopped")
    finally:
        httpd.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anonframework", description="anonframework results tools"
    )
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("serve", help="serve a results directory and open the dashboard")
    p.add_argument(
        "dir",
        nargs="?",
        default=DEFAULT_RESULTS_DIR,
        help=f"results directory to serve (default: {DEFAULT_RESULTS_DIR})",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="preferred port (default: 8000)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve(args.dir, host=args.host, port=args.port, open_browser=not args.no_browser)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
