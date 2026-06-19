"""
teaching.py — Phase 6+: TeachingScene

Works with both KnowledgeGraph (edges are Edge objects with .dst_id)
and SceneGraph (edges are (src_id, dst_id, label, relation) tuples).

Public API
----------
scene = TeachingScene(graph)
scene.next_step()
scene.prev_step()
scene.set_order([id1, …])
scene.update(dt, W, H)
scene.camera_offset()      → (dx, dy)
scene.current_node
scene.step_index
scene.total_steps
scene.active
scene.toggle()
scene.start(screen_w, screen_h)
"""

import math


def _lerp(a, b, t):
    return a + (b - a) * t


PAN_SMOOTH = 0.07


class TeachingScene:
    """Manages focus / dim state and camera pan for a teaching walkthrough."""

    def __init__(self, graph):
        self.graph     = graph
        self._order    = []
        self._step     = 0
        self.active    = False

        self._cam_x    = 0.0
        self._cam_y    = 0.0
        self._target_x = 0.0
        self._target_y = 0.0


    def set_order(self, node_ids):
        valid = [nid for nid in node_ids
                 if self.graph.get_node(nid) is not None]
        self._order = valid
        self._step  = 0
        self._apply_focus()

    def next_step(self):
        if not self._order: return
        self._step = min(self._step + 1, len(self._order) - 1)
        self._apply_focus()


    def next(self): self.next_step()

    def prev_step(self):
        if not self._order: return
        self._step = max(self._step - 1, 0)
        self._apply_focus()


    def prev(self): self.prev_step()

    def toggle(self):
        self.active = not self.active
        if self.active:
            if not self._order:
                self._build_default_order()
        else:
            self._clear_all_states()

    def start(self, screen_w=1280, screen_h=720):
        if not self._order:
            self._build_default_order()
        self.active = True
        self._step  = 0
        node = self.current_node
        if node:
            self._target_x = screen_w / 2.0 - node.x3d
            self._target_y = screen_h / 2.0 - node.y3d
        self._apply_focus()

    def stop(self):
        self.active = False
        self._clear_all_states()

    def sync_graph(self, graph):
        self.graph  = graph
        self._order = []
        self._step  = 0


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

    @property
    def step_label(self):
        if not self._order: return ""
        return f"{self._step + 1} / {len(self._order)}"


    def update(self, dt, screen_w, screen_h):
        if not self.active:
            self._target_x = _lerp(self._target_x, 0.0, 0.04)
            self._target_y = _lerp(self._target_y, 0.0, 0.04)
        else:
            node = self.current_node
            if node:
                self._target_x = screen_w / 2.0 - node.x3d
                self._target_y = screen_h / 2.0 - node.y3d

        t = min(1.0, PAN_SMOOTH * (dt * 60))
        self._cam_x = _lerp(self._cam_x, self._target_x, t)
        self._cam_y = _lerp(self._cam_y, self._target_y, t)

    def camera_offset(self):
        return self._cam_x, self._cam_y


    def draw_hud(self, img, W, H):
        import cv2
        C_TEAL  = (220, 240,  20)
        C_AMBER = ( 30, 190, 255)
        C_DIM   = ( 40,  40,  40)

        if not self.active:
            return
        node = self.current_node
        if node is None:
            return
        bx, by = W // 2 - 200, H - 80
        ov = img.copy()
        cv2.rectangle(ov, (bx, by), (bx + 400, by + 60), (10, 20, 15), -1)
        cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (bx, by), (bx + 400, by + 60), C_TEAL, 1, cv2.LINE_AA)

        title = node.title if hasattr(node, 'title') else (node.label if hasattr(node, 'label') else "")
        body  = node.body[:60] if hasattr(node, 'body') else ""
        cv2.putText(img, title, (bx + 12, by + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEAL, 1, cv2.LINE_AA)
        cv2.putText(img, body, (bx + 12, by + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 200, 140), 1, cv2.LINE_AA)
        cv2.putText(img, self.step_label, (bx + 350, by + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_AMBER, 1, cv2.LINE_AA)
        cv2.putText(img, "N=next  P=prev  T=stop teach",
                    (bx + 60, by + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, C_DIM, 1, cv2.LINE_AA)


    def _apply_focus(self):
        if not self.active: return
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

    def _dst_ids(self, nid):
        """Extract destination IDs from edges regardless of graph type."""
        dst_ids = []
        for edge in self.graph.edges:
            if isinstance(edge, tuple):

                if edge[0] == nid:
                    dst_ids.append(edge[1])
            else:

                if edge.src_id == nid:
                    dst_ids.append(edge.dst_id)
        return dst_ids

    def _build_default_order(self):
        nodes = self.graph.nodes
        if not nodes: return
        all_ids = {n.id for n in nodes}
        has_parent = set()
        for edge in self.graph.edges:
            if isinstance(edge, tuple):
                has_parent.add(edge[1])
            else:
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
                for child in self._dst_ids(nid):
                    if child not in visited:
                        visited.add(child)
                        nxt.append(child)
            queue = nxt
        for nid in all_ids:
            if nid not in visited:
                order.append(nid)
        self._order = order
