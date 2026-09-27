"""Tests for the ``anonframework serve`` CLI."""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

import pytest

from anonframework import cli


def test_ensure_dashboard_writes_html(tmp_path: Path) -> None:
    assert cli.ensure_dashboard(tmp_path) is True
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "<" in html and len(html) > 1000


def test_looks_like_results(tmp_path: Path) -> None:
    assert cli.looks_like_results(tmp_path) is False
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert cli.looks_like_results(tmp_path) is True


def test_make_server_falls_back_when_port_taken(tmp_path: Path) -> None:
    first = cli.make_server("127.0.0.1", 0, tmp_path)
    taken = first.server_address[1]
    try:
        second = cli.make_server("127.0.0.1", taken, tmp_path)  # port busy -> ephemeral
        try:
            assert second.server_address[1] != taken
        finally:
            second.server_close()
    finally:
        first.server_close()


def test_serve_missing_dir_returns_2(tmp_path: Path) -> None:
    assert cli.serve(str(tmp_path / "does-not-exist")) == 2


def test_serve_actually_serves_the_tree(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"schema_version": 4}', encoding="utf-8")
    cli.ensure_dashboard(tmp_path)
    httpd = cli.make_server("127.0.0.1", 0, tmp_path)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        page = urllib.request.urlopen(f"{base}/dashboard.html", timeout=5).read()
        assert b"<" in page
        manifest = urllib.request.urlopen(f"{base}/manifest.json", timeout=5).read()
        assert b"schema_version" in manifest
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_main_dispatches_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_serve(directory: str, **kw: object) -> int:
        seen["dir"] = directory
        seen["open_browser"] = kw.get("open_browser")
        seen["port"] = kw.get("port")
        return 0

    monkeypatch.setattr(cli, "serve", fake_serve)
    rc = cli.main(["serve", "some/dir", "--no-browser", "--port", "9123"])
    assert rc == 0
    assert seen == {"dir": "some/dir", "open_browser": False, "port": 9123}


def test_main_no_command_prints_help() -> None:
    assert cli.main([]) == 1
