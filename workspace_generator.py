"""
workspace_generator.py — Phase 7: AI Workspace Generation

WorkspaceGenerator is the successor to SceneGenerator. Instead of always
producing a generic knowledge graph, it first decides *what kind of
spatial structure* best represents the request — a workflow, a system
architecture, a hierarchy, a timeline, a flat list, an iterative process,
or a free-form concept map — and then emits nodes/edges tagged with that
intent, laid out accordingly.

This is a thin layer on top of the existing object model: it does not
require the full WorkspaceObject/Container/FlowStep class hierarchy from
the Phase 7 spec to be built yet. Every generated item is still a
SpatialNode and every relationship is still a graph.Edge — but each one
now carries a `kind` / `relation` tag (stored in SpatialNode.metadata and
Edge.relation, both pre-existing extension points) so the richer object
types and distinct per-relation visual styles described in the spec can
be layered on later without another data-migration pass.

Usage
-----
    from workspace_generator import WorkspaceGenerator
    from graph import KnowledgeGraph

    gen   = WorkspaceGenerator()   # or WorkspaceGenerator(host=..., model=...)
    graph = KnowledgeGraph()
    gen.generate(prompt="Design a customer onboarding workflow", graph=graph,
                 screen_w=1280, screen_h=720)
    print(gen.last_structure_type)   # e.g. "workflow"

Requires Ollama (https://ollama.com) running and reachable at `host`
(default http://localhost:11434), with the chosen model pulled, e.g.:

    ollama pull llama3.1

Async / threaded use
--------------------
WorkspaceGenerator.generate_async(prompt, graph, callback, screen_w, screen_h)
runs the HTTP call in a background daemon thread so the AR loop never stalls.
`callback(new_nodes, error)` is invoked on completion (error is None on
success), matching SceneGenerator's callback contract so call sites can
swap one generator for the other.
"""

import json
import os
import threading
import urllib.error
import urllib.request
from collections import defaultdict

from node import SpatialNode

# ── Default Ollama connection ───────────────────────────────────────
OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
REQUEST_TIMEOUT = 60

# ── Vocabulary ───────────────────────────────────────────────────────
STRUCTURE_TYPES = {
    "workflow", "architecture", "hierarchy",
    "timeline", "list", "process", "concept_map",
}

NODE_KINDS = {
    "idea_node", "flow_step", "decision",
    "container", "code_block", "note", "timeline_event",
}

RELATIONS = {
    "contains", "depends_on", "causes", "connects_to",
    "calls", "creates", "follows", "blocks",
}

# kind → (category, icon) — maps onto node.py's existing icon vocabulary
# (bulb, tree, brackets, function, scroll, cpu, flow, dot) so today's
# renderer already draws something sensible for every new kind.
KIND_VISUALS = {
    "idea_node":      ("concept",   "bulb"),
    "flow_step":       ("process",   "flow"),
    "decision":         ("concept",   "function"),
    "container":        ("structure", "tree"),
    "code_block":       ("code",      "brackets"),
    "note":             ("concept",   "scroll"),
    "timeline_event":   ("history",   "scroll"),
}
_DEFAULT_KIND = "idea_node"

# ── Layout tuning ────────────────────────────────────────────────────
TIMELINE_SPACING_X = 260
TIMELINE_Y_FRAC     = 0.45
LIST_SPACING_Y      = 150
LIST_X_FRAC          = 0.5
CONTAIN_SPACING_X   = 150
CONTAIN_OFFSET_Y    = 130

# ── System prompt ────────────────────────────────────────────────────
_SYSTEM = """
You are a spatial-thinking architect for Aether, an AR workspace that
turns ideas into manipulable 3D objects rather than static diagrams.

Before generating anything, decide the single best spatial representation
for the request. Choose exactly one "structure_type":

- "workflow"     a process with a clear start and an ordered sequence of steps
- "architecture" a system made of components and their dependencies
- "hierarchy"    a tree of parent/child concepts
- "timeline"     events ordered chronologically
- "list"         a flat set of related items with no strong structure between them
- "process"      a sequence of steps that branches or loops (like a workflow, but iterative)
- "concept_map"  a free-form web of related ideas — use this when nothing else fits

Return ONLY valid JSON — no prose, no markdown, no code fences.

JSON schema:
{
  "topic": "string — the main topic",
  "structure_type": one of ["workflow","architecture","hierarchy","timeline","list","process","concept_map"],
  "nodes": [
    {
      "id": "n1",
      "title": "short title (<=25 chars)",
      "body": "one-sentence description (<=90 chars)",
      "kind": one of ["idea_node","flow_step","decision","container","code_block","note","timeline_event"],
      "order": integer sequence position (only for workflow/process/timeline/list; omit otherwise),
      "parent": "id of a container-kind node this item belongs inside (omit if none)"
    }
  ],
  "edges": [
    {
      "from": "n1",
      "to": "n2",
      "relation": one of ["contains","depends_on","causes","connects_to","calls","creates","follows","blocks"],
      "label": "short relation text (<=12 chars, optional — defaults to the relation name)"
    }
  ]
}

Guidelines:
- 4 to 10 nodes per scene (more is visually cluttered).
- Use "decision" only for genuine branch points, with 2+ outgoing edges
  whose labels distinguish the branches (e.g. "yes" / "no").
- Use "container" for a node that groups others: every child must set
  "parent" to the container's id AND there must be a "contains" edge
  from the container to each child.
- For workflow / process / timeline / list, set "order" on every node
  (0,1,2,...) reflecting sequence, and connect consecutive steps with
  relation "follows".
- For architecture / hierarchy, prefer "depends_on" or "calls" for
  dependencies and "contains" for composition.
- The first node in the array should be the root or starting point.
- Do not add any text outside the JSON object.
"""


def _build_payload(prompt: str, model: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.4},
    }


def _parse_response(raw_json: str):
    """
    Parse the model's JSON and return (nodes_data, edges_data, topic, structure_type).
    Strips accidental markdown fences; falls back to "concept_map" for an
    invalid/missing structure_type rather than failing the whole generation.
    """
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
    """
    Sequence node IDs for linear layouts (timeline / list / workflow fallback).
    Prefers explicit metadata['order']; otherwise walks the graph breadth-first
    from root nodes (no incoming edges), which matches the JSON array order
    the model was asked to produce.
    """
    nodes = graph.nodes
    if not nodes:
        return []

    if any(n.metadata.get("order") is not None for n in nodes):
        return [n.id for n in sorted(
            nodes,
            key=lambda n: (n.metadata.get("order") is None, n.metadata.get("order", 0)),
        )]

    all_ids    = [n.id for n in nodes]
    has_parent = {e.dst_id for e in graph.edges}
    roots      = [nid for nid in all_ids if nid not in has_parent] or [all_ids[0]]

    visited = set(roots)
    order   = []
    queue   = list(roots)
    while queue:
        nxt = []
        for nid in queue:
            order.append(nid)
            for child in graph._adj.get(nid, []):
                if child not in visited:
                    visited.add(child)
                    nxt.append(child)
        queue = nxt
    for nid in all_ids:
        if nid not in visited:
            order.append(nid)
    return order


def _layout_timeline(graph, screen_w, screen_h):
    ids = _ordered_ids(graph)
    if not ids:
        return
    total_w = (len(ids) - 1) * TIMELINE_SPACING_X
    start_x = screen_w / 2.0 - total_w / 2.0
    y       = screen_h * TIMELINE_Y_FRAC
    for i, nid in enumerate(ids):
        node = graph.get_node(nid)
        node.x3d = start_x + i * TIMELINE_SPACING_X
        node.y3d = y
        node.z3d = 0.0


def _layout_list(graph, screen_w, screen_h):
    ids = _ordered_ids(graph)
    if not ids:
        return
    total_h = (len(ids) - 1) * LIST_SPACING_Y
    start_y = max(120.0, screen_h / 2.0 - total_h / 2.0)
    x       = screen_w * LIST_X_FRAC
    for i, nid in enumerate(ids):
        node = graph.get_node(nid)
        node.x3d = x
        node.y3d = start_y + i * LIST_SPACING_Y
        node.z3d = 0.0


def _nest_containers(graph):
    """
    Pull "contains" children into a tidy row beneath their container so the
    grouping reads visually even though Container isn't yet a distinct
    rendered object type — that's later WorkspaceObject-model work.
    """
    children_by_parent = defaultdict(list)
    for edge in graph.edges:
        if edge.relation == "contains":
            children_by_parent[edge.src_id].append(edge.dst_id)

    for parent_id, child_ids in children_by_parent.items():
        parent = graph.get_node(parent_id)
        if not parent:
            continue
        total_w = (len(child_ids) - 1) * CONTAIN_SPACING_X
        start_x = parent.x3d - total_w / 2.0
        y       = parent.y3d + CONTAIN_OFFSET_Y
        for i, cid in enumerate(child_ids):
            child = graph.get_node(cid)
            if child:
                child.x3d = start_x + i * CONTAIN_SPACING_X
                child.y3d = y
                child.z3d = parent.z3d


def _layout_for_structure(graph, structure_type, screen_w, screen_h):
    if structure_type == "timeline":
        _layout_timeline(graph, screen_w, screen_h)
    elif structure_type == "list":
        _layout_list(graph, screen_w, screen_h)
    else:
        # workflow / process / architecture / hierarchy / concept_map all
        # read well with the existing BFS-level + force-directed layout:
        # a pure chain renders as a vertical line, branches fan out
        # naturally, and unrelated clusters settle apart from each other.
        graph.auto_layout(screen_w, screen_h)

    # Containment is a stronger spatial constraint than the general
    # layout pass, so it always gets the final say regardless of structure.
    _nest_containers(graph)


def _populate_graph(nodes_data, edges_data, structure_type, graph, screen_w, screen_h):
    id_map = {}

    for i, nd in enumerate(nodes_data):
        nid  = nd.get("id") or f"n{i}"
        kind = nd.get("kind")
        if kind not in NODE_KINDS:
            kind = _DEFAULT_KIND
        category, icon = KIND_VISUALS[kind]

        node = SpatialNode(
            title        = nd.get("title", "?"),
            body         = nd.get("body",  ""),
            category     = category,
            icon         = icon,
            ai_generated = True,
        )
        node.metadata["kind"]           = kind
        node.metadata["structure_type"] = structure_type

        order = nd.get("order")
        if isinstance(order, (int, float)):
            node.metadata["order"] = int(order)

        parent_id = nd.get("parent")
        if parent_id:
            node.metadata["parent"] = parent_id

        graph.add_node(node)
        id_map[nid] = node

    for ed in edges_data:
        src = id_map.get(ed.get("from", ""))
        dst = id_map.get(ed.get("to",   ""))
        if not src or not dst:
            continue
        relation = ed.get("relation")
        if relation not in RELATIONS:
            relation = "connects_to"
        label = ed.get("label") or relation.replace("_", " ")
        graph.connect(src, dst, label, relation)

    # Backfill a "contains" edge for any explicit parent/child pair the
    # model declared via "parent" but forgot to also wire as an edge.
    for nd in nodes_data:
        parent_id = nd.get("parent")
        child     = id_map.get(nd.get("id"))
        parent    = id_map.get(parent_id)
        if parent and child and not graph.get_edge(parent.id, child.id):
            graph.connect(parent, child, "contains", "contains")

    _layout_for_structure(graph, structure_type, screen_w, screen_h)

    return list(id_map.values())


class WorkspaceGenerator:
    """
    Calls a local Ollama server, classifies the best spatial structure for
    the request, and populates a KnowledgeGraph accordingly.

    Parameters
    ----------
    host : str, optional
        Base URL of the Ollama server. Falls back to OLLAMA_HOST env var,
        then "http://localhost:11434".
    model : str, optional
        Model name (must already be pulled). Falls back to OLLAMA_MODEL
        env var, then "llama3.1".
    """

    def __init__(self, host=None, model=None):
        self._host = (host or OLLAMA_HOST).rstrip("/")
        self._model = model or OLLAMA_MODEL
        self.last_structure_type = None
        self.last_topic = None

    def _call_api(self, prompt: str) -> str:
        url     = f"{self._host}/api/chat"
        payload = json.dumps(_build_payload(prompt, self._model)).encode()
        req     = urllib.request.Request(
            url,
            data    = payload,
            method  = "POST",
            headers = {"Content-Type": "application/json"},
        )
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

        text = body.get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("No content in Ollama response")
        return text

    def generate(self, prompt: str, graph, screen_w=1280, screen_h=720):
        """
        Synchronously generate a workspace from `prompt` and add it to `graph`.
        Returns the list of newly created SpatialNode objects.
        Raises RuntimeError on any failure (network, bad JSON, etc).
        """
        raw = self._call_api(prompt)
        nodes_data, edges_data, topic, structure_type = _parse_response(raw)
        self.last_structure_type = structure_type
        self.last_topic = topic
        return _populate_graph(nodes_data, edges_data, structure_type,
                                graph, screen_w, screen_h)

    def generate_async(self, prompt: str, graph,
                        callback=None, screen_w=1280, screen_h=720):
        """
        Generate a workspace in a background thread.
        `callback(new_nodes, error)` is called when done; `error` is None
        on success, or an Exception on failure. Same contract as
        SceneGenerator.generate_async, so call sites can swap generators.
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