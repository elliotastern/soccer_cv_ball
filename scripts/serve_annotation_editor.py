#!/usr/bin/env python3
"""
Standalone server for editing COCO bounding boxes in an annotation folder.
Serves the folder (images + _annotations.coco.json) and an editor UI.
Run from project root; open http://localhost:PORT/editor.html
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Project root (soccer_cv_ball)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FOLDER = PROJECT_ROOT / "data" / "raw" / "prelabelled_balls"
DEFAULT_PORT = 8004


def main():
    parser = argparse.ArgumentParser(description="Serve annotation folder for editing COCO bounding boxes")
    parser.add_argument("--folder", type=str, default=str(DEFAULT_FOLDER), help="Annotation folder (images + _annotations.coco.json)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to serve on")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)
    coco_path = folder / "_annotations.coco.json"
    if not coco_path.exists():
        print(f"Error: {coco_path} not found")
        sys.exit(1)

    editor_html_path = Path(__file__).parent / "coco_annotation_editor.html"
    if not editor_html_path.exists():
        print(f"Error: Editor HTML not found: {editor_html_path}")
        sys.exit(1)

    # Use a simple HTTP server that serves folder + editor + save endpoint
    import http.server
    import socketserver

    folder_str = str(folder.resolve())
    editor_html = editor_html_path.read_text(encoding="utf-8")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=folder_str, **kwargs)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/editor.html" or path == "/":
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(editor_html.encode("utf-8"))
                return
            # Everything else: serve from annotation folder (images, _annotations.coco.json)
            super().do_GET()

        def do_POST(self):
            if self.path != "/save":
                self.send_response(404)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Not found"}).encode())
                return
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length == 0:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "No content"}).encode())
                    return
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
                out_path = Path(folder_str) / "_annotations.coco.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "message": f"Saved to {out_path}"}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())

        def log_message(self, format, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", args.port), Handler)
    print("=" * 60)
    print("Annotation folder editor (COCO bounding boxes)")
    print("=" * 60)
    print(f"Folder: {folder}")
    print(f"URL:    http://localhost:{args.port}/editor.html")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
