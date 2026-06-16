"""
teaching.py — Phase 6: TeachingScene

A TeachingScene wraps a KnowledgeGraph and adds a linear presentation
layer: the user (or the AI) steps through nodes one at a time.  The
focused node is highlighted (amber); all others are dimmed.  A "camera"
smoothly pans the entire scene so the focused node is centred.

Public API
----------
scene = TeachingScene(graph)
scene.next_step()          # advance to the next node in the sequence
scene.prev_step()          # go back
scene.set_order([id1, …])  # set the presentation order explicitly
scene.update(dt, W, H)     # call each frame to animate pan
scene.apply_camera(node)   # transform node coords by current camera pan
scene.current_node         # currently focused SpatialNode (or None)
scene.step_index           # 0-based index
scene.total_steps          # total in sequence
scene.active               # bool — teaching mode on/off
scene.toggle()
"""

import math


def _lerp(a, b, t):
    return a + (b - a) * t


# Camera pan smoothing
PAN_SMOOTH = 0.07   # fraction per frame (EMA)


class TeachingScene:
    """Manages focus / dim state and camera pan for a teaching walkthrough."""

    def __init__(self, graph):
        self.graph       = graph
        self._order      = []      # list of node IDs in presentation order
        self._step       = 0
        self.active      = False

        # Virtual camera pan offset (world-px)
        self._cam_x      = 0.0
        self._cam_y      = 0.0
        self._target_x   = 0.0
        self._target_y   = 0.0

    # ── Sequence control ─────────────────────────────────────────────

    def set_order(self, node_ids):
        """Set the presentation order as a list of node IDs."""
        # Filter to only IDs that exist in the graph
        valid = [nid for nid in node_ids
                 if self.graph.get_node(nid) is not None]
        self._order = valid
        self._step  = 0
        self._apply_focus()

    def next_step(self):
        if not self._order:
            return
        self._step = min(self._step + 1, len(self._order) - 1)
        self._apply_focus()

    def prev_step(self):
        if not self._order:
            return
        self._step = max(self._step - 1, 0)
        self._apply_focus()

    def toggle(self):
        self.active = not self.active
        if self.active:
            # Default order: BFS from first node with no incoming edges
            if not self._order:
                self._build_default_order()
        else:
            self._clear_all_states()

    def start(self, screen_w=1280, screen_h=720):
        """Enter teaching mode and jump to step 0."""
        if not self._order:
            self._build_default_order()
        self.active  = True
        self._step   = 0
        # Centre camera on first node
        node = self.current_node
        if node:
            self._target_x = screen_w / 2.0 - node.x3d
            self._target_y = screen_h / 2.0 - node.y3d
        self._apply_focus()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def current_node(self):
        if not self._order or self._step >= len(self._order):
            return None
        return self.graph.get_node(self._order[self._step])

    @property
    def step_index(self):
        return self._step

    @property
    def total_steps(self):
        return len(self._order)

    # ── Update (call each frame) ──────────────────────────────────────

    def update(self, dt, screen_w, screen_h):
        """
        Smooth camera pan toward the focused node.
        Does NOT modify node world positions — callers apply the pan
        offset only for rendering via apply_camera().
        """
        if not self.active:
            # Relax camera back to identity
            self._target_x = _lerp(self._target_x, 0.0, 0.04)
            self._target_y = _lerp(self._target_y, 0.0, 0.04)
        else:
            node = self.current_node
            if node:
                # Target: centre the focused node on screen
                self._target_x = screen_w / 2.0 - node.x3d
                self._target_y = screen_h / 2.0 - node.y3d

        t = min(1.0, PAN_SMOOTH * (dt * 60))
        self._cam_x = _lerp(self._cam_x, self._target_x, t)
        self._cam_y = _lerp(self._cam_y, self._target_y, t)

    def camera_offset(self):
        """Return (dx, dy) world-space pan offset to add to node positions."""
        return self._cam_x, self._cam_y

    # ── Internal ─────────────────────────────────────────────────────

    def _apply_focus(self):
        if not self.active:
            return
        current_id = (self._order[self._step]
                      if self._order and self._step < len(self._order)
                      else None)
        for node in self.graph.nodes:
            node.focused = (node.id == current_id)
            node.dimmed  = (node.id != current_id)

    def _clear_all_states(self):
        for node in self.graph.nodes:
            node.focused = False
            node.dimmed  = False

    def _build_default_order(self):
        """BFS order starting from roots."""
        nodes = self.graph.nodes
        if not nodes:
            return
        # Use the graph's adjacency directly
        all_ids   = {n.id for n in nodes}
        has_parent = set()
        for edge in self.graph.edges:
            has_parent.add(edge.dst_id)
        roots = [nid for nid in all_ids if nid not in has_parent]
        if not roots:
            roots = [nodes[0].id]

        visited = set()
        queue   = list(roots)
        order   = []
        for r in roots:
            visited.add(r)
        while queue:
            nxt = []
            for nid in queue:
                order.append(nid)
                for child in self.graph._adj.get(nid, []):
                    if child not in visited:
                        visited.add(child)
                        nxt.append(child)
            queue = nxt
        # Add any disconnected nodes at the end
        for nid in all_ids:
            if nid not in visited:
                order.append(nid)
        self._order = order