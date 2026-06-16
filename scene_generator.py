"""
scene_generator.py — Phase 6: AI Scene Generation

SceneGenerator takes a natural-language prompt such as
  "Teach me binary trees"
and returns a populated KnowledgeGraph of SpatialNode objects, with edges,
by calling the Anthropic Messages API.

The generator uses a structured JSON protocol so the response can be parsed
reliably without any external dependencies beyond the stdlib `json` module.

Usage
-----
    from scene_generator import SceneGenerator
    from graph import KnowledgeGraph

    gen   = SceneGenerator(api_key="sk-ant-…")   # or omit for env-var ANTHROPIC_API_KEY
    graph = KnowledgeGraph()
    gen.generate(prompt="Teach me recursion", graph=graph,
                 screen_w=1280, screen_h=720)

The graph is populated in-place; nodes and edges are added, existing content
is not touched.

Async / threaded use
--------------------
SceneGenerator.generate_async(prompt, graph, callback, screen_w, screen_h)
runs the HTTP call in a background daemon thread so the AR loop never stalls.
`callback(graph, error)` is invoked on completion (may be None on success).
"""

import json
import os
import threading
import urllib.error
import urllib.request
from node import SpatialNode

# ── Default model & endpoint ──────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-sonnet-4-6"
MAX_TOKENS        = 2048

# ── System prompt ─────────────────────────────────────────────────────
_SYSTEM = """
You are a spatial knowledge architect for Aether, an AR learning environment.
Given a topic or question, return a JSON object describing a knowledge graph.

Return ONLY valid JSON — no prose, no markdown, no code fences.

The JSON schema is:
{
  "topic": "string — the main topic",
  "nodes": [
    {
      "id": "n1",
      "title": "short title (≤25 chars)",
      "body": "one-sentence description (≤90 chars)",
      "category": one of ["concept","structure","code","math","history","system","process"],
      "icon": one of ["bulb","tree","brackets","function","scroll","cpu","flow","dot"]
    }
  ],
  "edges": [
    {"from": "n1", "to": "n2", "label": "short relation (≤12 chars, optional)"}
  ]
}

Guidelines:
- 4 to 8 nodes per scene (more is visually cluttered).
- Edges should form a meaningful directed acyclic graph (tree is fine).
- The first node in the array should be the root concept.
- Keep titles and bodies concise — they will be drawn on 230×138 px cards.
- Category and icon should match the node's nature (e.g. code → brackets).
- Do not add any text outside the JSON object.
"""


def _build_payload(prompt: str) -> dict:
    return {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "system":     _SYSTEM,
        "messages":   [{"role": "user", "content": prompt}],
    }


def _parse_response(raw_json: str):
    """
    Parse the API response and return (nodes_data, edges_data, topic).
    Strips accidental markdown fences if present.
    """
    text = raw_json.strip()
    # Strip ```json … ``` wrappers the model sometimes adds despite instructions
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    data = json.loads(text)
    return data.get("nodes", []), data.get("edges", []), data.get("topic", "")


def _populate_graph(nodes_data, edges_data, graph, screen_w, screen_h):
    """
    Turn parsed JSON into SpatialNode objects inside `graph`.
    Returns the list of newly added SpatialNode objects.
    """
    id_map = {}   # JSON id string → SpatialNode

    for nd in nodes_data:
        node = SpatialNode(
            title        = nd.get("title", "?"),
            body         = nd.get("body",  ""),
            category     = nd.get("category", "concept"),
            icon         = nd.get("icon", None),
            ai_generated = True,
        )
        graph.add_node(node)
        id_map[nd["id"]] = node

    for ed in edges_data:
        src = id_map.get(ed.get("from", ""))
        dst = id_map.get(ed.get("to",   ""))
        if src and dst:
            graph.connect(src, dst, ed.get("label", ""))

    # Auto-layout immediately so nodes appear in a sensible arrangement
    graph.auto_layout(screen_w, screen_h)

    return list(id_map.values())


class SceneGenerator:
    """
    Calls the Anthropic API and populates a KnowledgeGraph.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key.  Falls back to the ANTHROPIC_API_KEY env var.
    """

    def __init__(self, api_key=None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def _call_api(self, prompt: str) -> str:
        """Synchronous HTTP call; returns the raw text from the model."""
        payload = json.dumps(_build_payload(prompt)).encode()
        req     = urllib.request.Request(
            ANTHROPIC_API_URL,
            data    = payload,
            method  = "POST",
            headers = {
                "Content-Type":      "application/json",
                "x-api-key":         self._api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"API HTTP {e.code}: {e.read().decode()[:200]}") from e

        # Extract text from the content blocks
        for block in body.get("content", []):
            if block.get("type") == "text":
                return block["text"]
        raise RuntimeError("No text block in API response")

    def generate(self, prompt: str, graph, screen_w=1280, screen_h=720):
        """
        Synchronously generate a scene from `prompt` and add it to `graph`.
        Returns the list of newly created SpatialNode objects.
        Raises RuntimeError on any failure.
        """
        raw      = self._call_api(prompt)
        n, e, _t = _parse_response(raw)
        return _populate_graph(n, e, graph, screen_w, screen_h)

    def generate_async(self, prompt: str, graph,
                       callback=None, screen_w=1280, screen_h=720):
        """
        Generate a scene in a background thread.
        `callback(new_nodes, error)` is called when done; `error` is None
        on success, or an Exception on failure.
        """
        def _worker():
            try:
                nodes = self.generate(prompt, graph, screen_w, screen_h)
                if callback:
                    callback(nodes, None)
            except Exception as exc:
                if callback:
                    callback([], exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t