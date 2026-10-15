"""Start a temporary API, ingest synthetic PDF, and verify a cited answer.

Requires DATABASE_URL pointing at a prepared disposable demo database.
Use --mock-ollama for a deterministic plumbing check, NOT real LLM validation.
"""

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryFile
from threading import Thread
import time

from dotenv import load_dotenv
import pymupdf
import requests


TEXT = (
    "The fictional Cedar clinic opens at nine each morning and its appointment "
    "desk closes at five every weekday. This synthetic example is not clinical guidance."
)
QUESTION = "When does the fictional Cedar clinic open?"


def make_pdf() -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), TEXT)
        document.set_metadata({"title": "Synthetic Cedar clinic demo"})
        return document.tobytes()


class MockOllama(BaseHTTPRequestHandler):
    """Test fixture only; deliberately distinct model identity from real vectors."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embeddings":
            if payload["model"] != "demo-mock-768":
                self.send_error(400)
                return
            # An explicit test fixture, not a claim of semantic model quality.
            relevant = "cedar" in payload["prompt"].lower()
            vector = ([1.0, 0.0] if relevant else [0.0, 1.0]) + [0.0] * 766
            body = {"embedding": vector}
        elif self.path == "/api/chat":
            sources = json.loads(payload["messages"][1]["content"])["sources"]
            body = {
                "message": {
                    "content": json.dumps(
                        {"quotes": [{"source_id": 1, "quote": sources[0]["text"]}]}
                    )
                }
            }
        else:
            self.send_error(404)
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def mock_ollama():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOllama)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_demo(environment: dict):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-access-log",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("API exited during startup")
                try:
                    if requests.get(f"{base}/health", timeout=1).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("API readiness timed out")
            uploaded = requests.post(
                f"{base}/documents",
                params={"file_name": "synthetic-demo.pdf", "version": "demo-v1"},
                data=make_pdf(),
                headers={"Content-Type": "application/pdf"},
                timeout=180,
            )
            uploaded.raise_for_status()
            answered = requests.post(
                f"{base}/questions", json={"question": QUESTION}, timeout=200
            )
            answered.raise_for_status()
            answer = answered.json()
            if answer["status"] != "answered" or not answer["citations"]:
                raise RuntimeError(
                    f"Expected a grounded answer, received {answer['status']}"
                )
            for citation in answer["citations"]:
                if (
                    citation["file_name"] != "synthetic-demo.pdf"
                    or citation["page"] != 1
                    or citation["quote"] not in TEXT
                ):
                    raise RuntimeError("Demo citation does not match the generated PDF")
            print(
                json.dumps(
                    {"upload": uploaded.json(), "question": QUESTION, "result": answer},
                    indent=2,
                )
            )
            print(
                "PASS: PDF -> API -> embeddings -> PostgreSQL -> retrieval -> cited answer"
            )
        except Exception:
            log.seek(0)
            # Uvicorn access logging is disabled; avoid printing request data.
            print(log.read(), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock-ollama",
        action="store_true",
        help="Run a labeled deterministic provider fixture",
    )
    args = parser.parse_args()
    load_dotenv()
    environment = os.environ.copy()
    if not environment.get("DATABASE_URL"):
        parser.error("Set DATABASE_URL to your prepared disposable demo database")
    environment["EMBEDDING_PROVIDER"] = "ollama"
    if args.mock_ollama:
        print(
            "MOCK PROVIDER DEMO: real HTTP/PDF/PostgreSQL, no actual LLM inference",
            flush=True,
        )
        with mock_ollama() as base:
            environment.update(
                OLLAMA_BASE_URL=base,
                OLLAMA_EMBEDDING_MODEL="demo-mock-768",
                OLLAMA_CHAT_MODEL="demo-mock-selector",
            )
            run_demo(environment)
    else:
        print(
            "LOCAL OLLAMA DEMO: requires downloaded embedding and chat models",
            flush=True,
        )
        run_demo(environment)


if __name__ == "__main__":
    main()
