"""
graph.py — Phase 8: Object graph + the connection-building framework.

  SceneGraph              owns objects + directed edges between them
  ConnectionRenderer       draws every edge with a style (solid / dashed /
                           pulse / directional) based on its relation
  ConnectionGestureSystem  implements the hands-only connection gesture:
                           point at object A, hold, drag a beam preview,
                           point at object B, release -> edge created.
"""

import cv2
import math
import time
from collections import defaultdict

from physics import Z_FAR_LIMIT

FOV = 900.0

CONNECT_HOLD_TIME   = 0.35
CONNECT_SNAP_RADIUS = 90
PULSE_SPEED = 0.45


RELATION_STYLE = {
    "data_flow":    ((220, 240,  20), False, True,  True),
    "dependency":   (( 60, 160, 220), True,  False, True),
    "control_flow": ((140, 220,  40), False, False, True),
    "grouping":     ((160, 160,  60), False, False, False),
    "hierarchy":    ((230,  90, 215), False, False, True),
}
DEFAULT_STYLE = ((180, 200, 50), False, True, True)


def _proj(x3d, y3d, z3d, scx, scy):
    s = FOV / max(FOV + z3d, 1.0)
    return scx + (x3d - scx) * s, scy + (y3d - scy) * s, s


class SceneGraph:
    def __init__(self):
        self._objects = {}
        self._edges   = {}
        self._adj     = defaultdict(set)
        self._radj    = defaultdict(set)

    def add_node(self, obj):
        self._objects[obj.id] = obj
        return obj

    def remove_node(self, obj_or_id):
        nid = obj_or_id if isinstance(obj_or_id, str) else obj_or_id.id
        if nid not in self._objects:
            return
        for dst in list(self._adj[nid]):
            self._edges.pop((nid, dst), None)
            self._radj[dst].discard(nid)
        for src in list(self._radj[nid]):
            self._edges.pop((src, nid), None)
            self._adj[src].discard(nid)
        self._adj.pop(nid, None)
        self._radj.pop(nid, None)
        self._objects.pop(nid, None)

    def get_node(self, nid):
        return self._objects.get(nid)

    @property
    def nodes(self):
        return list(self._objects.values())

    def connect(self, src, dst, relation="data_flow"):
        sid = src if isinstance(src, str) else src.id
        did = dst if isinstance(dst, str) else dst.id
        if sid not in self._objects or did not in self._objects or sid == did:
            return None
        if (sid, did) in self._edges:
            return self._edges[(sid, did)]
        self._edges[(sid, did)] = relation
        self._adj[sid].add(did)
        self._radj[did].add(sid)
        return relation

    def disconnect(self, src, dst):
        sid = src if isinstance(src, str) else src.id
        did = dst if isinstance(dst, str) else dst.id
        self._edges.pop((sid, did), None)
        self._adj[sid].discard(did)
        self._radj[did].discard(sid)

    @property
    def edges(self):
        return [(s, d, rel) for (s, d), rel in self._edges.items()]


class ConnectionRenderer:
    """Draws every edge in the graph with a style determined by relation."""

    def __init__(self):
        self._pulses = {}

    def render(self, img, graph, dt):
        H, W = img.shape[:2]
        scx, scy = W / 2.0, H / 2.0

        for src_id, dst_id, relation in graph.edges:
            src = graph.get_node(src_id)
            dst = graph.get_node(dst_id)
            if src is None or dst is None:
                continue

            color, dashed, pulse, arrow = RELATION_STYLE.get(relation, DEFAULT_STYLE)

            sx1, sy1, sc1 = _proj(src.x3d, src.y3d, src.z3d, scx, scy)
            sx2, sy2, _   = _proj(dst.x3d, dst.y3d, dst.z3d, scx, scy)

            mid_z = (src.z3d + dst.z3d) * 0.5
            fog_t = max(0.0, min(1.0, mid_z / max(Z_FAR_LIMIT, 1.0)))
            alpha = 1.0 - fog_t * 0.65
            col   = tuple(int(c * alpha) for c in color)

            either_active = src.selected or dst.selected or src.grabbed or dst.grabbed
            lw = 2 if either_active else 1

            if dashed:
                self._draw_dashed(img, (sx1, sy1), (sx2, sy2), col, lw)
            else:
                cv2.line(img, (int(sx1), int(sy1)), (int(sx2), int(sy2)), col, lw, cv2.LINE_AA)

            if arrow:
                self._draw_arrowhead(img, (sx1, sy1), (sx2, sy2), col, lw)

            if pulse:
                key = (src_id, dst_id)
                self._pulses[key] = (self._pulses.get(key, 0.0) + PULSE_SPEED * dt) % 1.0
                t = self._pulses[key]
                px = int(sx1 + (sx2 - sx1) * t)
                py = int(sy1 + (sy2 - sy1) * t)
                cv2.circle(img, (px, py), max(2, int(4 * sc1)), col, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed(img, p1, p2, color, lw, seg=10):
        x1, y1 = p1; x2, y2 = p2
        d = max(1.0, math.hypot(x2 - x1, y2 - y1))
        n = max(1, int(d // seg))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1.0, (i + 1) / n)
            a = (int(x1 + (x2 - x1) * t0), int(y1 + (y2 - y1) * t0))
            b = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
            cv2.line(img, a, b, color, lw, cv2.LINE_AA)

    @staticmethod
    def _draw_arrowhead(img, p1, p2, color, lw):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        d = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / d, dy / d
        tip_x, tip_y = x2 - ux * 26, y2 - uy * 26
        perp_x, perp_y = -uy, ux
        al = 13
        p_a = (int(tip_x - ux*al + perp_x*al*0.4), int(tip_y - uy*al + perp_y*al*0.4))
        p_b = (int(tip_x - ux*al - perp_x*al*0.4), int(tip_y - uy*al - perp_y*al*0.4))
        cv2.line(img, (int(tip_x), int(tip_y)), p_a, color, max(1, lw), cv2.LINE_AA)
        cv2.line(img, (int(tip_x), int(tip_y)), p_b, color, max(1, lw), cv2.LINE_AA)


class ConnectionGestureSystem:
    """
    Point at object A and hold CONNECT_HOLD_TIME -> dragging begins.
    While dragging, a beam preview follows the cursor.
    Stop pointing (release) over object B -> edge A->B created.
    Stop pointing over empty space -> cancelled.
    """

    def __init__(self, relation="data_flow"):
        self.relation   = relation
        self._src       = None
        self._hold_t0   = None
        self._dragging  = False

    @property
    def dragging(self):
        return self._dragging

    @property
    def source(self):
        return self._src

    def _find_target(self, cx, cy, objects, exclude=None):
        best_d, best_o = 9999.0, None
        for obj in objects:
            if obj is exclude:
                continue
            d = obj.screen_dist(cx, cy)
            if d < best_d:
                best_d, best_o = d, obj
        return best_o if best_d < CONNECT_SNAP_RADIUS else None

    def update(self, right_hand, cx, cy, objects, graph, t):
        """Returns the new (src, dst) tuple if a connection was just made."""
        if not right_hand.visible or not right_hand.is_pointing:
            result = None
            if self._dragging:
                dst = self._find_target(cx, cy, objects, exclude=self._src)
                if dst is not None and self._src is not None:
                    graph.connect(self._src, dst, self.relation)
                    result = (self._src, dst)
            self._src = None
            self._hold_t0 = None
            self._dragging = False
            return result

        if self._dragging:
            return None

        target = self._find_target(cx, cy, objects)
        if target is None:
            self._hold_t0 = None
            return None

        if self._hold_t0 is None:
            self._hold_t0 = t
            self._src = target
            return None

        if (t - self._hold_t0) >= CONNECT_HOLD_TIME:
            self._dragging = True
        return None

    def draw_preview(self, img, cx, cy):
        if not self._dragging or self._src is None:
            return
        H, W = img.shape[:2]
        scx, scy = W / 2.0, H / 2.0
        sx, sy, _ = _proj(self._src.x3d, self._src.y3d, self._src.z3d, scx, scy)
        p1 = (int(sx), int(sy))
        p2 = (int(cx), int(cy))
        glow = img.copy()
        cv2.line(glow, p1, p2, (220, 240, 20), 6, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.25, img, 0.75, 0, img)
        cv2.line(img, p1, p2, (220, 240, 20), 2, cv2.LINE_AA)
        cv2.circle(img, p2, 6, (220, 240, 20), 2, cv2.LINE_AA)