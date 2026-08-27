#!/usr/bin/env python3
"""Serve the slide editor and persist feedback to a normal JSON file."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FEEDBACK_PATH = ROOT / "slide-feedback.json"


def empty_feedback() -> dict:
    return {"version": 1, "updatedAt": None, "feedback": {}, "catalog": {}}


def read_feedback() -> dict:
    if not FEEDBACK_PATH.exists():
        return empty_feedback()
    try:
        saved = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_feedback()
    if not isinstance(saved, dict):
        return empty_feedback()
    saved.setdefault("version", 1)
    saved.setdefault("updatedAt", None)
    saved.setdefault("feedback", {})
    saved.setdefault("catalog", {})
    return saved


def write_feedback(payload: dict) -> dict:
    feedback = payload.get("feedback", {})
    catalog = payload.get("catalog", {})
    if not isinstance(feedback, dict) or not isinstance(catalog, dict):
        raise ValueError("feedback and catalog must be objects")
    saved = {
        "version": 1,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "feedback": feedback,
        "catalog": catalog,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=ROOT, prefix=".slide-feedback-", suffix=".json"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            json.dump(saved, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
        os.replace(temporary_name, FEEDBACK_PATH)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return saved


class SlideHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/api/feedback":
            self.send_json(read_feedback())
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/feedback":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2_000_000:
                raise ValueError("payload too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            saved = write_feedback(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json(saved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SlideHandler)
    print(f"Slide editor: http://{args.host}:{args.port}/slides.html", flush=True)
    print(f"Feedback file: {FEEDBACK_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
