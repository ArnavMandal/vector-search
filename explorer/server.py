"""Local search explorer. Start from the project root with make explorer."""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from vector_search import Index

ROOT = Path(__file__).resolve().parent
ASSETS = {
    "/": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    "/explorer.js": ("explorer.js", "text/javascript"),
}
SEED = 42
K = 10
M = 3
EF_CONSTRUCTION = 16
EF_RANGE = (10, 100)
DOMAIN = (100, 64)
INITIAL_QUERY = (55, 31)
INITIAL_EF = 12


def make_points():
    rng = np.random.default_rng(SEED)
    centers = [(18, 44), (31, 17), (51, 43), (71, 24), (83, 49)]
    clouds = [rng.normal(center, (6, 4), size=(44, 2)) for center in centers]
    background = rng.uniform((2, 2), (98, 62), size=(80, 2))
    points = np.vstack([*clouds, background]).astype("f4")
    np.clip(points, (1, 1), (99, 63), out=points)
    rng.shuffle(points)
    return points


def parse_query(search):
    params = parse_qs(search, keep_blank_values=True)
    if set(params) != {"x", "y", "ef"} or any(len(v) != 1 for v in params.values()):
        raise ValueError("Supply one x, y and ef value.")
    x, y = float(params["x"][0]), float(params["y"][0])
    ef = int(params["ef"][0])
    if not (math.isfinite(x) and 0 <= x <= DOMAIN[0] and math.isfinite(y) and 0 <= y <= DOMAIN[1]):
        raise ValueError("Query coordinates must lie inside the map.")
    if not EF_RANGE[0] <= ef <= EF_RANGE[1]:
        raise ValueError("Search breadth is outside the slider range.")
    return np.array([x, y], dtype="f4"), ef


class ExplorerServer(ThreadingHTTPServer):
    def __init__(self, port):
        points = make_points()
        self.index = Index(points, hnsw=True, M=M, efConstruction=EF_CONSTRUCTION)
        self.dataset = dict(points=points.tolist(), domain=DOMAIN, seed=SEED, k=K, M=M,
                            efConstruction=EF_CONSTRUCTION, efRange=EF_RANGE,
                            initialQuery=INITIAL_QUERY, initialEf=INITIAL_EF)
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    def reply(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Moving the slider can cancel the previous request.

    def send_json(self, value, status=200):
        self.reply(json.dumps(value, allow_nan=False).encode(), "application/json", status)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path in ASSETS:
            filename, content_type = ASSETS[url.path]
            self.reply((ROOT / filename).read_bytes(), content_type)
        elif url.path == "/api/dataset":
            self.send_json(self.server.dataset)
        elif url.path == "/api/search":
            try:
                query, ef = parse_query(url.query)
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
                return
            graph = self.server.index.trace_hnsw(query, K, efSearch=ef)
            exact_ids, distances = self.server.index.search(query[None, :], K)
            self.send_json(dict(query=query.tolist(), ef=ef, graph=graph,
                                exact=exact_ids[0].tolist(), radius=math.sqrt(distances[0, -1])))
        else:
            self.send_error(404)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    with ExplorerServer(parser.parse_args().port) as server:
        print(f"Search explorer: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
