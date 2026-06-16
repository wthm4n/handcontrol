"""
graph.py — Phase 6: KnowledgeGraph

Manages a set of SpatialNode objects and directed edges between them.
Provides:
  • KnowledgeGraph    — add/remove nodes & edges, auto-layout, serialisation
  • ConnectionRenderer — animated connection lines with directional arrows,
                         depth-aware alpha, and animated flow pulses
"""

import cv2
import math
import time
from collections import defaultdict

# ── Layout constants ──────────────────────────────────────────────────
LAYOUT_PADDING_X  = 340   # horizontal gap between nodes (world-px)
LAYOUT_PADDING_Y  = 200   # vertical gap between tree levels
LAYOUT_ROOT_Y     = 180   # screen Y for root node
LAYOUT_FORCE_K    = 0.04  # spring constant for force-directed layout
LAYOUT_REPULSE_R  = 320   # ideal separation (world-px)
LAYOUT_DAMP       = 0.55  # velocity damping per step
LAYOUT_STEPS      = 80    # relaxation iterations for auto-layout

# ── Connection rendering constants ────────────────────────────────────
CONN_LINE_COLOR  = (160, 160, 60)   # BGR — soft cyan-ish
CONN_PULSE_COLOR = (220, 240, 20)   # BGR — bright teal for the traveling dot
CONN_ARROW_LEN   = 14               # pixels — arrowhead length
CONN_PULSE_SPEED = 0.45             # 0..1 of edge length per second
FOV              = 900.0


def _lerp(a, b, t):
    return a + (b - a) * t


def _dist2d(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _project(x3d, y3d, z3d, screen_cx, screen_cy):
    scale = FOV / max(FOV + z3d, 1.0)
    sx = screen_cx + (x3d - screen_cx) * scale
    sy = screen_cy + (y3d - screen_cy) * scale
    return int(sx), int(sy), scale


class Edge:
    """A directed edge from node_a → node_b with an optional label."""

    def __init__(self, src_id, dst_id, label="", relation="connects_to"):
        self.src_id = src_id
        self.dst_id = dst_id
        self.label  = label
        # Semantic relationship kind — one of: contains, depends_on,
        # causes, connects_to, calls, creates, follows, blocks.
        # Renderer-agnostic for now (ConnectionRenderer still draws every
        # edge the same way); this is forward-compatible storage for the
        # distinct visual styles described in the Phase 7 spec.
        self.relation = relation
        # Animated pulse position (0..1 along the edge, wraps around)
        self._pulse_t = 0.0

    @property
    def key(self):
        return (self.src_id, self.dst_id)

    def tick(self, dt):
        self._pulse_t = (self._pulse_t + CONN_PULSE_SPEED * dt) % 1.0


class KnowledgeGraph:
    """
    Source of truth for all nodes and edges in the workspace.

    Nodes are just SpatialNode objects; the graph owns their positions
    when auto-layout is requested.

    Typical lifecycle
    -----------------
    kg = KnowledgeGraph()
    n1 = kg.add_node(SpatialNode("Recursion", body="…"))
    n2 = kg.add_node(SpatialNode("Base Case", body="…"))
    kg.connect(n1, n2, "has")
    kg.auto_layout(screen_w=1280, screen_h=720)
    """

    def __init__(self):
        self._nodes = {}          # id → SpatialNode
        self._edges = {}          # (src_id, dst_id) → Edge
        self._adj   = defaultdict(set)   # src_id → {dst_id}
        self._radj  = defaultdict(set)   # dst_id → {src_id}

    # ── Nodes ─────────────────────────────────────────────────────────

    def add_node(self, node):
        """Register a SpatialNode. Returns the node for convenience."""
        self._nodes[node.id] = node
        # Sync connection list on the node itself
        node.connections = list(self._adj.get(node.id, set()))
        return node

    def remove_node(self, node_or_id):
        nid = node_or_id if isinstance(node_or_id, str) else node_or_id.id
        if nid not in self._nodes:
            return
        # Remove all edges touching this node
        for dst in list(self._adj[nid]):
            self._edges.pop((nid, dst), None)
            self._radj[dst].discard(nid)
        for src in list(self._radj[nid]):
            self._edges.pop((src, nid), None)
            self._adj[src].discard(nid)
        self._adj.pop(nid, None)
        self._radj.pop(nid, None)
        self._nodes.pop(nid, None)

    def get_node(self, node_id):
        return self._nodes.get(node_id)

    @property
    def nodes(self):
        return list(self._nodes.values())

    # ── Edges ─────────────────────────────────────────────────────────

    def connect(self, src, dst, label="", relation="connects_to"):
        """Add a directed edge src → dst. Accepts nodes or node IDs.
        Returns the Edge object (existing one if already connected)."""
        sid = src if isinstance(src, str) else src.id
        did = dst if isinstance(dst, str) else dst.id
        if sid not in self._nodes or did not in self._nodes:
            return None
        if (sid, did) in self._edges:
            return self._edges[(sid, did)]  # already connected
        edge = Edge(sid, did, label, relation)
        self._edges[edge.key] = edge
        self._adj[sid].add(did)
        self._radj[did].add(sid)
        # Update node connection caches
        self._nodes[sid].connections = list(self._adj[sid])
        self._nodes[did].connections = list(self._adj[did])
        return edge

    def get_edge(self, src, dst):
        """Look up the Edge object between src and dst (nodes or IDs), or None."""
        sid = src if isinstance(src, str) else src.id
        did = dst if isinstance(dst, str) else dst.id
        return self._edges.get((sid, did))

    def disconnect(self, src, dst):
        sid = src if isinstance(src, str) else src.id
        did = dst if isinstance(dst, str) else dst.id
        key = (sid, did)
        if key not in self._edges:
            return
        del self._edges[key]
        self._adj[sid].discard(did)
        self._radj[did].discard(sid)

    @property
    def edges(self):
        return list(self._edges.values())

    # ── Auto-layout ───────────────────────────────────────────────────

    def auto_layout(self, screen_w=1280, screen_h=720, root_id=None):
        """
        Hierarchical + force-directed layout.

        1. Identify roots (nodes with no incoming edges) — or use root_id.
        2. BFS to assign tree levels.
        3. Assign initial positions from the tree structure.
        4. Run force-directed relaxation to un-overlap.
        """
        nodes = self.nodes
        if not nodes:
            return

        cx = screen_w / 2.0
        node_ids = [n.id for n in nodes]

        # ── Step 1: find roots ────────────────────────────────────────
        if root_id and root_id in self._nodes:
            roots = [root_id]
        else:
            roots = [nid for nid in node_ids
                     if not self._radj.get(nid)]
            if not roots:
                roots = [node_ids[0]]

        # ── Step 2: BFS levels ────────────────────────────────────────
        level = {}
        queue = list(roots)
        for r in roots:
            level[r] = 0
        visited = set(roots)
        while queue:
            nxt = []
            for nid in queue:
                for child in self._adj.get(nid, []):
                    if child not in visited:
                        visited.add(child)
                        level[child] = level[nid] + 1
                        nxt.append(child)
            queue = nxt

        # Nodes unreachable from roots get their own level
        max_lvl = max(level.values(), default=0) + 1
        for nid in node_ids:
            if nid not in level:
                level[nid] = max_lvl

        # ── Step 3: initial positions ────────────────────────────────
        by_level = defaultdict(list)
        for nid, lvl in level.items():
            by_level[lvl].append(nid)

        pos = {}
        for lvl, group in by_level.items():
            total_w = (len(group) - 1) * LAYOUT_PADDING_X
            start_x = cx - total_w / 2.0
            y = LAYOUT_ROOT_Y + lvl * LAYOUT_PADDING_Y
            for i, nid in enumerate(group):
                pos[nid] = [start_x + i * LAYOUT_PADDING_X, y]

        # ── Step 4: force-directed relaxation ────────────────────────
        vels = {nid: [0.0, 0.0] for nid in node_ids}
        for _ in range(LAYOUT_STEPS):
            forces = {nid: [0.0, 0.0] for nid in node_ids}

            # Repulsion between all pairs
            for i, a in enumerate(node_ids):
                for b in node_ids[i + 1:]:
                    dx = pos[a][0] - pos[b][0]
                    dy = pos[a][1] - pos[b][1]
                    d  = max(1.0, math.sqrt(dx * dx + dy * dy))
                    if d < LAYOUT_REPULSE_R:
                        mag = LAYOUT_FORCE_K * (LAYOUT_REPULSE_R - d)
                        nx_, ny_ = dx / d, dy / d
                        forces[a][0] += nx_ * mag
                        forces[a][1] += ny_ * mag
                        forces[b][0] -= nx_ * mag
                        forces[b][1] -= ny_ * mag

            # Spring attraction along edges
            for edge in self.edges:
                a, b = edge.src_id, edge.dst_id
                if a not in pos or b not in pos:
                    continue
                dx = pos[b][0] - pos[a][0]
                dy = pos[b][1] - pos[a][1]
                d  = max(1.0, math.sqrt(dx * dx + dy * dy))
                mag = LAYOUT_FORCE_K * (d - LAYOUT_PADDING_X)
                nx_, ny_ = dx / d, dy / d
                forces[a][0] += nx_ * mag
                forces[a][1] += ny_ * mag
                forces[b][0] -= nx_ * mag
                forces[b][1] -= ny_ * mag

            # Integrate
            for nid in node_ids:
                vels[nid][0] = (vels[nid][0] + forces[nid][0]) * LAYOUT_DAMP
                vels[nid][1] = (vels[nid][1] + forces[nid][1]) * LAYOUT_DAMP
                pos[nid][0] += vels[nid][0]
                pos[nid][1] += vels[nid][1]

        # Write back to nodes
        for nid, (x, y) in pos.items():
            n = self._nodes[nid]
            n.x3d = max(100, min(screen_w - 100, x))
            n.y3d = max(80,  min(screen_h * 0.85, y))
            n.z3d = 0.0

    # ── Clustering ────────────────────────────────────────────────────

    def cluster_by_category(self, screen_w=1280, screen_h=720):
        """
        Group nodes by category, placing each cluster in a distinct
        screen region before running a local relaxation.
        """
        from collections import Counter
        cats = Counter(n.category for n in self.nodes)
        cat_list = list(cats.keys())
        nc = max(1, len(cat_list))

        cx, cy = screen_w / 2.0, screen_h / 2.0
        radius = min(cx, cy) * 0.65

        cat_centres = {}
        for i, cat in enumerate(cat_list):
            angle = 2 * math.pi * i / nc - math.pi / 2
            cat_centres[cat] = (
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle),
            )

        # Scatter nodes within their category cluster
        import random
        rng = random.Random(42)
        for node in self.nodes:
            ccx, ccy = cat_centres[node.category]
            node.x3d = ccx + rng.uniform(-120, 120)
            node.y3d = ccy + rng.uniform(-80,   80)
            node.z3d = 0.0

        # Light relaxation
        self.auto_layout(screen_w, screen_h)


# ── Connection Renderer ───────────────────────────────────────────────

class ConnectionRenderer:
    """
    Draws animated edges between nodes onto a frame.

    Call render() once per frame after all nodes have been updated.
    """

    def render(self, img, graph, dt):
        """
        img   : output BGR frame
        graph : KnowledgeGraph
        dt    : frame delta-time (seconds)
        """
        H, W = img.shape[:2]
        scx, scy = W / 2.0, H / 2.0
        now = time.time()

        for edge in graph.edges:
            src = graph.get_node(edge.src_id)
            dst = graph.get_node(edge.dst_id)
            if src is None or dst is None:
                continue

            edge.tick(dt)

            # Project both endpoints
            sx1, sy1, sc1 = _project(src.x3d, src.y3d, src.z3d, scx, scy)
            sx2, sy2, sc2 = _project(dst.x3d, dst.y3d, dst.z3d, scx, scy)

            # Midpoint depth for fog
            mid_z    = (src.z3d + dst.z3d) * 0.5
            from physics import Z_FAR_LIMIT
            fog_t    = max(0.0, min(1.0, mid_z / max(Z_FAR_LIMIT, 1.0)))
            alpha    = 1.0 - fog_t * 0.65

            # Base line color
            base_col = CONN_LINE_COLOR
            col = tuple(int(c * alpha) for c in base_col)
            if col == (0, 0, 0):
                continue

            # Line thickness: thicker for highlighted edges
            either_sel = src.selected or dst.selected or src.grabbed or dst.grabbed
            lw = 2 if either_sel else 1

            cv2.line(img, (sx1, sy1), (sx2, sy2), col, lw, cv2.LINE_AA)

            # Edge label (midpoint)
            if edge.label:
                mx, my = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                label_col = tuple(int(c * alpha * 0.7) for c in base_col)
                cv2.putText(img, edge.label, (mx + 4, my - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.26, label_col,
                            1, cv2.LINE_AA)

            # Arrowhead at dst
            dx = sx2 - sx1
            dy = sy2 - sy1
            d = max(1.0, math.sqrt(dx * dx + dy * dy))
            ux, uy = dx / d, dy / d  # unit vector
            # Arrow tip sits just before dst centre (so it doesn't overlap the card)
            tip_x = sx2 - int(ux * 28)
            tip_y = sy2 - int(uy * 28)
            perp_x, perp_y = -uy, ux
            al = CONN_ARROW_LEN
            p1 = (int(tip_x - ux * al + perp_x * al * 0.4),
                  int(tip_y - uy * al + perp_y * al * 0.4))
            p2 = (int(tip_x - ux * al - perp_x * al * 0.4),
                  int(tip_y - uy * al - perp_y * al * 0.4))
            arrow_col = tuple(int(c * alpha * 0.9) for c in CONN_PULSE_COLOR)
            cv2.line(img, (tip_x, tip_y), p1, arrow_col, max(1, lw), cv2.LINE_AA)
            cv2.line(img, (tip_x, tip_y), p2, arrow_col, max(1, lw), cv2.LINE_AA)

            # Animated pulse dot traveling along the edge
            t = edge._pulse_t
            px = int(sx1 + (sx2 - sx1) * t)
            py = int(sy1 + (sy2 - sy1) * t)
            pulse_r  = max(2, int(4 * sc1))
            pulse_col = tuple(int(c * alpha) for c in CONN_PULSE_COLOR)
            cv2.circle(img, (px, py), pulse_r, pulse_col, -1, cv2.LINE_AA)