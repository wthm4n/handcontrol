"""
workspace_generator.py — Phase 7+: AI Workspace Generation

Generates physically-intuitive 3D object scenes rather than flat cards.

  workflow / process   → row of PipeSegments
  hierarchy / tree     → TreeSphere nodes with branch edges
  list / array         → row of ArrayBlocks
  architecture         → TreeSphere network
  timeline             → ArrayBlocks along a horizontal line
  loop / iterator      → LoopMarker
  stack / queue        → StackDisc column
  bitmask / flags      → BitField
  concept_map          → ArrayBlocks (fallback)
"""

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict

from objects import (
    ArrayBlock, TreeSphere, PipeSegment, StackDisc,
    LoopMarker, BitField, SceneGraph
)

OLLAMA_HOST    = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
REQUEST_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "240"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

STRUCTURE_TYPES = {
    "workflow", "architecture", "hierarchy",
    "timeline", "list", "process", "concept_map",
    "stack", "loop", "bitfield",
}

NODE_KINDS = {
    "idea_node", "flow_step", "decision",
    "container", "code_block", "note", "timeline_event",
    "stack_frame", "loop_counter", "bit_group",
}

RELATIONS = {
    "contains", "depends_on", "causes", "connects_to",
    "calls", "creates", "follows", "blocks",
}


KIND_SHAPE = {
    "idea_node":      (ArrayBlock,   "concept"),
    "flow_step":       (PipeSegment,  "process"),
    "decision":         (TreeSphere,   "concept"),
    "container":        (TreeSphere,   "structure"),
    "code_block":       (ArrayBlock,   "code"),
    "note":             (ArrayBlock,   "concept"),
    "timeline_event":   (ArrayBlock,   "history"),
    "stack_frame":      (StackDisc,    "structure"),
    "loop_counter":     (LoopMarker,   "concept"),
    "bit_group":        (BitField,     "code"),
}
_DEFAULT_KIND = "idea_node"

TIMELINE_SPACING_X = 260
TIMELINE_Y_FRAC     = 0.45
LIST_SPACING_Y      = 160
LIST_X_FRAC         = 0.5
CONTAIN_SPACING_X   = 200
CONTAIN_OFFSET_Y    = 140
STACK_SPACING_Y     = 90

_SYSTEM = """
You are a spatial-thinking architect for Aether, an AR workspace that
turns ideas into manipulable 3D objects.

Decide the single best spatial structure for the request. Choose from:
  "workflow"     ordered process steps → PipeSegments
  "architecture" component network     → TreeSpheres
  "hierarchy"    parent/child tree     → TreeSpheres
  "timeline"     chronological events  → ArrayBlocks in a line
  "list"         flat related items    → ArrayBlocks in a column
  "process"      branching/looping     → PipeSegments + LoopMarker
  "stack"        LIFO/FIFO structure   → StackDiscs stacked vertically
  "concept_map"  free-form web         → ArrayBlocks

Return ONLY valid JSON. No prose, no markdown, no code fences.

JSON schema:
{
  "topic": "string",
  "structure_type": one of the types above,
  "nodes": [
    {
      "id": "n1",
      "title": "<=25 chars",
      "body": "<=90 chars",
      "kind": one of ["idea_node","flow_step","decision","container","code_block",
                      "note","timeline_event","stack_frame","loop_counter","bit_group"],
      "order": integer (sequential position; omit if no order),
      "parent": "id of container node (omit if none)"
    }
  ],
  "edges": [
    {
      "from": "n1",
      "to": "n2",
      "relation": one of ["contains","depends_on","causes","connects_to",
                          "calls","creates","follows","blocks"],
      "label": "<=12 chars (optional)"
    }
  ]
}

Rules:
- 4–10 nodes. First node = root or start.
- workflow/process/timeline/list: set "order" on every node, connect with "follows".
- architecture/hierarchy: use "depends_on" or "calls" for dependencies.
- decision: 2+ outgoing edges with branch labels.
- No text outside the JSON object.
"""


def _build_payload(prompt, model):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.4},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }


def _parse_response(raw_json):
    text = raw_json.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    data = json.loads(text)
    structure_type = data.get("structure_type", "concept_map")
    if structure_type not in STRUCTURE_TYPES:
        structure_type = "concept_map"
    return (data.get("nodes", []), data.get("edges", []),
            data.get("topic", ""), structure_type)


def _ordered_ids(graph):
    nodes = graph.nodes
    if not nodes: return []
    if any(n.metadata.get("order") is not None for n in nodes):
        return [n.id for n in sorted(
            nodes,
            key=lambda n: (n.metadata.get("order") is None, n.metadata.get("order", 0)),
        )]
    all_ids    = [n.id for n in nodes]
    has_parent = {e[1] for e in graph.edges}
    roots      = [nid for nid in all_ids if nid not in has_parent] or [all_ids[0]]
    visited = set(roots); order = []; queue = list(roots)
    while queue:
        nxt = []
        for nid in queue:
            order.append(nid)
            for e in graph.edges:
                if e[0] == nid and e[1] not in visited:
                    visited.add(e[1]); nxt.append(e[1])
        queue = nxt
    for nid in all_ids:
        if nid not in visited: order.append(nid)
    return order


def _layout_timeline(graph, screen_w, screen_h):
    ids = _ordered_ids(graph)
    if not ids: return
    total_w = (len(ids) - 1) * TIMELINE_SPACING_X
    start_x = screen_w / 2.0 - total_w / 2.0
    y = screen_h * TIMELINE_Y_FRAC
    for i, nid in enumerate(ids):
        n = graph.get_node(nid)
        n.x3d = start_x + i * TIMELINE_SPACING_X
        n.y3d = y; n.z3d = 0.0


def _layout_list(graph, screen_w, screen_h):
    ids = _ordered_ids(graph)
    if not ids: return
    total_h = (len(ids) - 1) * LIST_SPACING_Y
    start_y = max(120.0, screen_h / 2.0 - total_h / 2.0)
    x = screen_w * LIST_X_FRAC
    for i, nid in enumerate(ids):
        n = graph.get_node(nid)
        n.x3d = x; n.y3d = start_y + i * LIST_SPACING_Y; n.z3d = 0.0


def _layout_stack(graph, screen_w, screen_h):
    """Stack discs vertically in order."""
    ids = _ordered_ids(graph)
    if not ids: return
    cx = screen_w / 2.0
    total_h = (len(ids) - 1) * STACK_SPACING_Y
    start_y = screen_h / 2.0 - total_h / 2.0
    for i, nid in enumerate(ids):
        n = graph.get_node(nid)
        n.x3d = cx; n.y3d = start_y + i * STACK_SPACING_Y; n.z3d = 0.0


def _nest_containers(graph):
    children_by_parent = defaultdict(list)
    for s_id, d_id, lbl, rel in graph.edges:
        if rel == "contains":
            children_by_parent[s_id].append(d_id)
    for parent_id, child_ids in children_by_parent.items():
        parent = graph.get_node(parent_id)
        if not parent: continue
        total_w = (len(child_ids) - 1) * CONTAIN_SPACING_X
        start_x = parent.x3d - total_w / 2.0
        y = parent.y3d + CONTAIN_OFFSET_Y
        for i, cid in enumerate(child_ids):
            child = graph.get_node(cid)
            if child:
                child.x3d = start_x + i * CONTAIN_SPACING_X
                child.y3d = y; child.z3d = parent.z3d


def _layout_for_structure(graph, structure_type, screen_w, screen_h):
    if structure_type == "timeline":
        _layout_timeline(graph, screen_w, screen_h)
    elif structure_type == "list":
        _layout_list(graph, screen_w, screen_h)
    elif structure_type == "stack":
        _layout_stack(graph, screen_w, screen_h)
    else:
        graph.auto_layout(screen_w, screen_h)
    _nest_containers(graph)


def _make_object(nd, kind, structure_type):
    """Instantiate the correct SpatialObject subclass for this node."""
    ShapeClass, category = KIND_SHAPE.get(kind, KIND_SHAPE[_DEFAULT_KIND])

    title = nd.get("title", "?")
    body  = nd.get("body",  "")

    kwargs = dict(label=title, category=category)

    if ShapeClass is ArrayBlock:
        obj = ArrayBlock(value=title, **kwargs)
        obj.body = body
    elif ShapeClass is TreeSphere:
        obj = TreeSphere(radius=30.0, **kwargs)
        obj.body = body
    elif ShapeClass is PipeSegment:
        obj = PipeSegment(pipe_w=85, pipe_h=48, **kwargs)
        obj.body = body
    elif ShapeClass is StackDisc:
        obj = StackDisc(disc_rx=62, disc_ry=20, **kwargs)
        obj.body = body
    elif ShapeClass is LoopMarker:
        obj = LoopMarker(orbit_r=52, orbit_speed=2.0, **kwargs)
        obj.body = body
    elif ShapeClass is BitField:
        obj = BitField(bits=8, value=0, **kwargs)
        obj.body = body
    else:
        obj = ArrayBlock(value=title, **kwargs)
        obj.body = body

    obj.ai_generated = True
    return obj


def _populate_graph(nodes_data, edges_data, structure_type, graph, screen_w, screen_h):
    id_map = {}

    for i, nd in enumerate(nodes_data):
        nid  = nd.get("id") or f"n{i}"
        kind = nd.get("kind")
        if kind not in NODE_KINDS:
            kind = _DEFAULT_KIND

        obj = _make_object(nd, kind, structure_type)
        obj.metadata["kind"]           = kind
        obj.metadata["structure_type"] = structure_type

        order = nd.get("order")
        if isinstance(order, (int, float)):
            obj.metadata["order"] = int(order)

        parent_id = nd.get("parent")
        if parent_id:
            obj.metadata["parent"] = parent_id

        graph.add_node(obj)
        id_map[nid] = obj

    for ed in edges_data:
        src = id_map.get(ed.get("from", ""))
        dst = id_map.get(ed.get("to",   ""))
        if not src or not dst: continue
        relation = ed.get("relation")
        if relation not in RELATIONS:
            relation = "connects_to"
        label = ed.get("label") or relation.replace("_", " ")
        graph.connect(src, dst, label, relation)


    for nd in nodes_data:
        parent_id = nd.get("parent")
        child  = id_map.get(nd.get("id"))
        parent = id_map.get(parent_id)
        if parent and child and not graph.get_edge(parent.id, child.id):
            graph.connect(parent, child, "contains", "contains")

    _layout_for_structure(graph, structure_type, screen_w, screen_h)
    return list(id_map.values())


class WorkspaceGenerator:
    """
    Calls a local Ollama server, classifies the best spatial structure,
    and populates a SceneGraph with physically-intuitive 3D objects.
    """

    def __init__(self, host=None, model=None):
        self._host  = (host or OLLAMA_HOST).rstrip("/")
        self._model = model or OLLAMA_MODEL
        self.last_structure_type = None
        self.last_topic = None

    def _call_api(self, prompt):
        url     = f"{self._host}/api/chat"
        payload = json.dumps(_build_payload(prompt, self._model)).encode()
        req     = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode()[:200]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host} "
                f"(is `ollama serve` running?): {e.reason}"
            ) from e
        except (socket.timeout, TimeoutError) as e:
            elapsed = time.monotonic() - started
            raise RuntimeError(
                f"Ollama didn't finish after {elapsed:.0f}s (limit {REQUEST_TIMEOUT}s). "
                f"Try: ollama run {self._model}  to warm it up, or raise OLLAMA_TIMEOUT."
            ) from e

        text = body.get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("No content in Ollama response")
        return text

    def generate(self, prompt, graph, screen_w=1280, screen_h=720):
        raw = self._call_api(prompt)
        nodes_data, edges_data, topic, structure_type = _parse_response(raw)
        self.last_structure_type = structure_type
        self.last_topic = topic
        return _populate_graph(nodes_data, edges_data, structure_type,
                                graph, screen_w, screen_h)

    def generate_async(self, prompt, graph,
                       callback=None, screen_w=1280, screen_h=720):
        def _worker():
            try:
                nodes = self.generate(prompt, graph, screen_w, screen_h)
                if callback: callback(nodes, None)
            except Exception as exc:
                if callback: callback([], exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t