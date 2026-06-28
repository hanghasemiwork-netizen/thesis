"""
wall_aware_paths.py
Multi-floor wall-aware grid for A* pathfinding.

Key design decisions:
- Space interiors eroded by 40cm — paths keep clearance from walls
- Door bridge polygons built at full width (0.55m) — never eroded
- Stair landing discs (0.80m radius) added AFTER erosion — always passable
- Columns subtracted with 40cm clearance buffer
- Stair bounding box subtracted as hard obstacle (shrunk 10cm inward)
- Nested space subtraction prevents cross-space shortcuts
"""

import numpy as np
import heapq
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union, nearest_points

GRID_SIZE = 0.15  # 15cm cells


class FloorGrid:
    """Single-floor occupancy grid with A* pathfinding."""

    def __init__(self, spaces, doors, space_boundaries,
                 stairs, columns, elevation_m, label=""):
        self.elevation_m = elevation_m
        self.label       = label
        self.walkable    = None
        self._space_polys_cleaned = {}
        self._build(spaces, doors, space_boundaries, stairs, columns)

    def _build(self, spaces, doors, space_boundaries, stairs, columns):
        all_x, all_y = [], []
        raw_polys    = {}

        # 1. Space footprints on this floor
        for space in spaces:
            elev = space.get("storey_elevation_m", 0.0)
            if abs(elev - self.elevation_m) > 1.0:
                continue
            fp = space.get("footprint", [])
            if not fp or len(fp) < 3:
                continue
            try:
                poly = Polygon(fp)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.area > 0.05:
                    raw_polys[space["id"]] = poly
                    all_x.extend(p[0] for p in fp)
                    all_y.extend(p[1] for p in fp)
            except Exception:
                pass

        if not all_x:
            self._empty()
            return

        # 2. Nested space subtraction
        ids           = list(raw_polys.keys())
        cleaned_polys = {}
        for sid in ids:
            poly = raw_polys[sid]
            for other_id in ids:
                if other_id == sid:
                    continue
                other = raw_polys[other_id]
                if poly.area <= other.area or not poly.contains(other.centroid):
                    continue
                try:
                    if poly.intersection(other).area / other.area > 0.80:
                        poly = poly.difference(other)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                except Exception:
                    pass
            cleaned_polys[sid] = poly
        self._space_polys_cleaned = cleaned_polys

        # 3. Erode space interiors by 40cm — paths keep clearance from walls
        eroded_spaces = []
        for sid, poly in cleaned_polys.items():
            try:
                e = poly.buffer(-0.40)
                if e.is_empty or e.area < poly.area * 0.10:
                    e = poly.buffer(-0.20)   # fallback for small spaces
                if e.is_empty or e.area < poly.area * 0.05:
                    e = poly                  # absolute fallback
                eroded_spaces.append(e)
            except Exception:
                eroded_spaces.append(poly)

        # 4. Door bridge polygons at FULL width — never eroded
        bridge_polys = []
        spaces_dict  = {s["id"]: s for s in spaces}
        doors_dict   = {d["id"]: d for d in doors}
        for space_id, door_id in space_boundaries:
            space = spaces_dict.get(space_id)
            door  = doors_dict.get(door_id)
            if not space or not door:
                continue
            if abs(space.get("storey_elevation_m", 0.0) - self.elevation_m) > 1.0:
                continue
            sp_poly = cleaned_polys.get(space_id)
            if sp_poly is None:
                continue
            dloc = door.get("location")
            if not dloc:
                continue
            try:
                d_pt      = Point(dloc[0], dloc[1])
                p_edge, _ = nearest_points(sp_poly, d_pt)
                line      = LineString([p_edge, d_pt])
                bridge_polys.append(line.buffer(0.55))
                bridge_polys.append(d_pt.buffer(0.40))
            except Exception:
                pass

        # 5. Stair landing discs — keep landings passable after space erosion
        stair_discs = []
        for stair in stairs:
            for loc_key in ["top_loc", "bottom_loc"]:
                sloc = stair.get(loc_key)
                if not sloc:
                    continue
                if abs(sloc[2] - self.elevation_m) > 1.5:
                    continue
                stair_discs.append(Point(sloc[0], sloc[1]).buffer(0.80))

        # Union: eroded spaces + full-width bridges + stair discs
        all_walkable = unary_union(eroded_spaces + bridge_polys + stair_discs)

        # 6. Subtract obstacles: stair box + columns (with 40cm clearance)
        obstacles = []
        for stair in stairs:
            if "bounds" in stair:
                mn_x, mn_y, mx_x, mx_y = stair["bounds"]
                obstacles.append(box(mn_x+0.10, mn_y+0.10, mx_x-0.10, mx_y-0.10))
        for col in columns:
            if abs(col.get("storey_elevation_m", 0.0) - self.elevation_m) > 1.0:
                continue
            fp = col.get("footprint", [])
            if not fp or len(fp) < 3:
                continue
            try:
                cp = Polygon(fp)
                if not cp.is_valid:
                    cp = cp.buffer(0)
                if cp.area > 0.01:
                    obstacles.append(cp.buffer(0.40))
            except Exception:
                pass

        if obstacles:
            try:
                wr = all_walkable.difference(unary_union(obstacles))
                all_walkable = wr if not wr.is_empty else all_walkable
            except Exception:
                pass

        # 7. Light cleanup
        try:
            c = all_walkable.buffer(-0.05).buffer(0.05)
            self.walkable = c if not c.is_empty else all_walkable
        except Exception:
            self.walkable = all_walkable

        # 8. Build grid
        self.x_min = min(all_x) - 1.0
        self.y_min = min(all_y) - 1.0
        x_max = max(all_x) + 1.0
        y_max = max(all_y) + 1.0
        self._cols = int((x_max - self.x_min) / GRID_SIZE) + 1
        self._rows = int((y_max - self.y_min) / GRID_SIZE) + 1
        self.grid  = np.zeros((self._rows, self._cols), dtype=np.uint8)

        for r in range(self._rows):
            for c in range(self._cols):
                wx, wy = self._w(r, c)
                if self.walkable.contains(Point(wx, wy)):
                    self.grid[r, c] = 1

        p = int(np.sum(self.grid))
        t = self._rows * self._cols
        print(f"[Grid] Floor {self.label} ({self.elevation_m}m): "
              f"{self._cols}x{self._rows} passable={p}/{t}")

    def _empty(self):
        self.grid  = np.zeros((1, 1), dtype=np.uint8)
        self.x_min = 0.0; self.y_min = 0.0
        self._cols = 1;   self._rows = 1
        self.walkable = None

    def _w(self, r, c):
        return (self.x_min + c * GRID_SIZE + GRID_SIZE / 2,
                self.y_min + r * GRID_SIZE + GRID_SIZE / 2)

    def _g(self, wx, wy):
        c = int((wx - self.x_min) / GRID_SIZE)
        r = int((wy - self.y_min) / GRID_SIZE)
        return (max(0, min(self._rows-1, r)),
                max(0, min(self._cols-1, c)))

    def nearest_passable(self, wx, wy):
        rc = self._g(wx, wy)
        if self.grid[rc] == 1:
            return rc
        for d in range(1, 40):
            for dr in range(-d, d+1):
                for dc in range(-d, d+1):
                    if abs(dr) != d and abs(dc) != d:
                        continue
                    nr, nc = rc[0]+dr, rc[1]+dc
                    if (0 <= nr < self._rows and 0 <= nc < self._cols
                            and self.grid[nr, nc] == 1):
                        return (nr, nc)
        return rc

    def astar(self, start_w, end_w):
        """Standard A* — walls, columns, stair box are hard obstacles."""
        if self.walkable is None:
            return None
        src = self.nearest_passable(start_w[0], start_w[1])
        dst = self.nearest_passable(end_w[0],   end_w[1])
        er, ec = dst

        def h(r, c):
            return np.hypot(r-er, c-ec) * GRID_SIZE

        heap = [(h(*src), 0.0, src[0], src[1], [src])]
        vis  = {}
        while heap:
            f, g, r, c, path = heapq.heappop(heap)
            if (r, c) in vis:
                continue
            vis[(r, c)] = g
            if r == er and c == ec:
                return self._thin([self._w(pr, pc) for pr, pc in path])
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < self._rows and 0 <= nc < self._cols):
                        continue
                    if self.grid[nr, nc] != 1 or (nr, nc) in vis:
                        continue
                    ng = g + np.hypot(dr, dc) * GRID_SIZE
                    heapq.heappush(heap,
                        (ng+h(nr, nc), ng, nr, nc, path+[(nr, nc)]))
        return None

    def _thin(self, pts, step=4):
        """Reduce waypoints — never skip if shortcut crosses a wall."""
        if not pts or len(pts) <= 2:
            return pts
        out = [pts[0]]; i = 0
        while i < len(pts) - 1:
            j = min(i+step, len(pts)-1)
            while j > i + 1:
                seg  = __import__("shapely.geometry", fromlist=["LineString"]).LineString([pts[i], pts[j]])
                diff = seg.difference(self.walkable)
                if diff.is_empty or diff.length < GRID_SIZE * 0.5:
                    break
                j -= 1
            out.append(pts[j]); i = j
        if out[-1] != pts[-1]:
            out.append(pts[-1])
        return out

    def path_distance(self, pts):
        if not pts or len(pts) < 2:
            return 0.0
        return sum(np.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
                   for i in range(len(pts)-1))


class GridMap:
    """Holds one FloorGrid per storey elevation."""

    def __init__(self, ifc_data):
        self.ifc_data = ifc_data
        self.floors   = {}
        self._build_all()

    def _build_all(self):
        spaces  = self.ifc_data["spaces"]
        doors   = self.ifc_data["doors"]
        sb      = self.ifc_data.get("space_boundaries", [])
        stairs  = self.ifc_data.get("stairs", [])
        columns = self.ifc_data.get("columns", [])
        storeys = self.ifc_data.get("storeys", [])

        elevations = sorted(set(
            round(s.get("storey_elevation_m", 0.0), 2)
            for s in spaces
        ))

        for elev in elevations:
            label = next(
                (st["name"] for st in storeys
                 if abs(st.get("elevation_m", 0.0) - elev) < 0.5),
                str(elev)
            )
            fg = FloorGrid(spaces, doors, sb, stairs, columns, elev, label)
            self.floors[elev] = fg

        print(f"[GridMap] Built {len(self.floors)} floor grid(s): "
              f"{list(self.floors.keys())}")

    def _get_floor(self, elev_hint=None):
        if not self.floors:
            return None
        if elev_hint is None:
            return self.floors[min(self.floors)]
        return self.floors[min(self.floors, key=lambda e: abs(e - elev_hint))]

    def astar(self, start_w, end_w, elev_hint=None):
        fg = self._get_floor(elev_hint)
        return fg.astar(start_w, end_w) if fg else None

    def path_distance(self, pts):
        if not pts or len(pts) < 2:
            return 0.0
        return sum(np.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
                   for i in range(len(pts)-1))

    def nearest_passable(self, wx, wy, elev_hint=None):
        fg = self._get_floor(elev_hint)
        return fg.nearest_passable(wx, wy) if fg else (0, 0)
