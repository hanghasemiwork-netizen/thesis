"""
path_finder.py
Greedy door-to-door evacuation path finder.

Logic per space:
  1. Build node list: doors from IfcRelSpaceBoundary + stair landings
     (stair landing added to a space if A* can reach it from that space)
  2. Pick best node = node closest (Euclidean) to FireExit
  3. Find farthest point in space footprint from best node
  4. A* (space-aware) from farthest point to best node
  5. If best node is stair → add stair dist + A* to FireExit. Done.
  6. If best node is regular door → greedy walk toward FireExit:
       - Try FireExit directly at each step
       - Try stair if on upper floor
       - Otherwise pick next door closest to FireExit

Walls, columns, stair bounding boxes = hard obstacles.
40cm clearance from walls/columns enforced by wall_aware_paths.py erosion.
"""

import numpy as np
import heapq
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

STAIR_TOP_ID = "__STAIR_TOP__"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dist(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])


def _snap(loc, fg):
    rc = fg.nearest_passable(loc[0], loc[1])
    return fg._w(*rc)


def _farthest_point(footprint, ref_loc, space_poly, fg):
    """Farthest footprint vertex from ref_loc that is inside the space
    and on a passable grid cell. Nudges toward centroid if on wall."""
    from wall_aware_paths import GRID_SIZE
    if not footprint or fg is None:
        return None
    cx = sum(p[0] for p in footprint) / len(footprint)
    cy = sum(p[1] for p in footprint) / len(footprint)
    candidates = sorted(footprint, key=lambda p: _dist(p, ref_loc), reverse=True)
    candidates = [(p[0], p[1]) for p in candidates] + [(cx, cy)]
    for raw in candidates:
        vx = cx-raw[0]; vy = cy-raw[1]; d = np.hypot(vx, vy); nudge = 0.0
        while nudge <= d + GRID_SIZE:
            tx = raw[0]+(vx/d)*nudge if d > 0 else cx
            ty = raw[1]+(vy/d)*nudge if d > 0 else cy
            rc = fg._g(tx, ty)
            if fg.grid[rc] == 1:
                pt = fg._w(*rc)
                if (space_poly is None
                        or space_poly.contains(Point(pt))
                        or space_poly.distance(Point(pt)) < GRID_SIZE * 2):
                    return pt
            nudge += GRID_SIZE
    return fg._w(*fg.nearest_passable(cx, cy))


def _astar_aware(fg, start, end, other_union, penalty=8.0):
    """A* with penalty for cells inside other spaces — keeps path in own space."""
    from wall_aware_paths import GRID_SIZE
    if fg.walkable is None:
        return None
    src = fg.nearest_passable(start[0], start[1])
    dst = fg.nearest_passable(end[0],   end[1])
    er, ec = dst
    def h(r, c): return np.hypot(r-er, c-ec) * GRID_SIZE
    heap = [(h(*src), 0.0, src[0], src[1], [src])]; vis = {}
    while heap:
        f, g, r, c, path = heapq.heappop(heap)
        if (r, c) in vis: continue
        vis[(r, c)] = g
        if r == er and c == ec:
            return fg._thin([fg._w(pr, pc) for pr, pc in path])
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r+dr, c+dc
                if not (0 <= nr < fg._rows and 0 <= nc < fg._cols): continue
                if fg.grid[nr, nc] != 1 or (nr, nc) in vis: continue
                step = np.hypot(dr, dc) * GRID_SIZE
                wx, wy = fg._w(nr, nc)
                pen = penalty if (other_union is not None
                                  and other_union.contains(Point(wx, wy))) else 0.0
                ng = g + step + pen
                heapq.heappush(heap, (ng+h(nr, nc), ng, nr, nc, path+[(nr, nc)]))
    return None


def _space_poly(space):
    fp = space.get("footprint", [])
    if not fp or len(fp) < 3: return None
    try:
        p = Polygon(fp); return p if p.is_valid else p.buffer(0)
    except Exception: return None


def _other_union(sid, elev, spaces):
    polys = []
    for o in spaces:
        if o["id"] == sid or abs(o.get("storey_elevation_m", 0)-elev) > 0.5: continue
        fp = o.get("footprint", [])
        if not fp or len(fp) < 3: continue
        try:
            p = Polygon(fp); polys.append(p if p.is_valid else p.buffer(0))
        except Exception: pass
    return unary_union(polys) if polys else None


# ── Main calculation ──────────────────────────────────────────────────────────

def calculate_all_paths(grid_map, ifc_data: dict) -> list:
    spaces  = ifc_data["spaces"]
    doors   = ifc_data["doors"]
    stairs  = ifc_data.get("stairs", [])
    sb_list = ifc_data.get("space_boundaries", [])

    doors_by_id = {d["id"]: d for d in doors}

    # ── Fire exit ─────────────────────────────────────────────────────────────
    fire_exit = next((d for d in doors if d.get("is_fire_exit")), None)
    if not fire_exit:
        print("[PathFinder] ERROR: No fire exit door found.")
        return []
    exit_loc  = (fire_exit["location"][0], fire_exit["location"][1])
    exit_elev = fire_exit.get("storey_elevation_m", 0.0)
    fg_exit   = grid_map._get_floor(exit_elev)

    # ── Stair setup ───────────────────────────────────────────────────────────
    stair_data       = stairs[0] if stairs else None
    stair_dist       = 0.0
    snapped_top      = None
    snapped_bot      = None
    stair_top_node   = None
    seg_bot_to_exit  = None
    dist_bot_to_exit = 0.0

    if stair_data:
        top_raw    = (stair_data["top_loc"][0],    stair_data["top_loc"][1])
        bot_raw    = (stair_data["bottom_loc"][0], stair_data["bottom_loc"][1])
        stair_dist = abs(stair_data["top_loc"][2] -
                         stair_data["bottom_loc"][2]) * 1.5
        fg_ff = grid_map._get_floor(stair_data["top_loc"][2])
        if fg_ff:
            snapped_top = fg_ff._w(*fg_ff.nearest_passable(top_raw[0], top_raw[1]))
        if fg_exit:
            snapped_bot      = fg_exit._w(*fg_exit.nearest_passable(bot_raw[0], bot_raw[1]))
            seg_bot_to_exit  = fg_exit.astar(snapped_bot, _snap(exit_loc, fg_exit))
            dist_bot_to_exit = fg_exit.path_distance(seg_bot_to_exit) if seg_bot_to_exit else 0.0
        if snapped_top:
            stair_top_node = {
                "id":                 STAIR_TOP_ID,
                "name":               "Stair_Top",
                "location":           (snapped_top[0], snapped_top[1],
                                       stair_data["top_loc"][2]),
                "storey_elevation_m": round(stair_data["top_loc"][2], 1),
                "width_m": 1.2, "height_m": 2.1, "angle_deg": 0.0,
                "is_fire_exit": False, "is_virtual": True,
            }

    # ── Build space→nodes from IfcRelSpaceBoundary ────────────────────────────
    space_to_nodes = {}
    for sp_id, d_id in sb_list:
        door = doors_by_id.get(d_id)
        if door:
            space_to_nodes.setdefault(sp_id, [])
            if door not in space_to_nodes[sp_id]:
                space_to_nodes[sp_id].append(door)

    # ── Add stair landing via GRID connectivity (general — no hardcoded dist) ─
    if stair_top_node and snapped_top:
        fg_stair = grid_map._get_floor(stair_data["top_loc"][2])
        if fg_stair:
            for space in spaces:
                elev = space.get("storey_elevation_m", 0.0)
                if abs(elev - stair_data["top_loc"][2]) > 1.0:
                    continue
                fp = space.get("footprint", [])
                if not fp or len(fp) < 3:
                    continue
                # Test from space centroid — can A* reach stair landing?
                cx = sum(p[0] for p in fp) / len(fp)
                cy = sum(p[1] for p in fp) / len(fp)
                test_start = fg_stair._w(*fg_stair.nearest_passable(cx, cy))
                seg = fg_stair.astar(test_start, snapped_top)
                if seg is not None:
                    space_to_nodes.setdefault(space["id"], [])
                    if stair_top_node not in space_to_nodes[space["id"]]:
                        space_to_nodes[space["id"]].append(stair_top_node)
                        print(f"[Stair] Grid-connected to: {space['name']}")

    all_doors_plus = doors + ([stair_top_node] if stair_top_node else [])

    print(f"[PathFinder] Calculating paths for {len(spaces)} spaces to 1 exit...")
    results = []

    for space in spaces:
        sid    = space["id"]
        elev   = space.get("storey_elevation_m", 0.0)
        fp     = space.get("footprint", [])
        fg     = grid_map._get_floor(elev)
        is_gnd = abs(elev - exit_elev) < 0.5

        nodes = space_to_nodes.get(sid, [])
        if not nodes or fg is None:
            results.append(_unreachable(space)); continue

        sp_poly = _space_poly(space)
        other_u = _other_union(sid, elev, spaces)

        # Best node = closest to FireExit
        directly_to_exit = fire_exit in nodes
        best_node = min(nodes,
                        key=lambda d: _dist((d["location"][0], d["location"][1]),
                                           exit_loc))
        best_loc = _snap((best_node["location"][0], best_node["location"][1]), fg)
        ref      = (best_node["location"][0], best_node["location"][1])

        # Farthest point from best node
        start_pt = _farthest_point(fp, ref, sp_poly, fg)
        if start_pt is None:
            results.append(_unreachable(space)); continue

        # Direct to exit
        if directly_to_exit and is_gnd and best_node["id"] == fire_exit["id"]:
            seg = fg.astar(start_pt, _snap(exit_loc, fg))
            if seg:
                dist = round(fg.path_distance(seg), 2)
                results.append({**_base(space),
                                 "total_distance_m": dist,
                                 "art57_distance_m": dist,
                                 "art61_distance_m": 0.0,
                                 "segments": [("ground", seg)],
                                 "reachable": True})
                continue

        # A* to best node (space-aware)
        seg_to_node = (_astar_aware(fg, start_pt, best_loc, other_u)
                       or fg.astar(start_pt, best_loc))
        if seg_to_node is None:
            results.append(_unreachable(space)); continue

        dist_to_node = fg.path_distance(seg_to_node)

        # If best node IS the stair → handle immediately
        if best_node["id"] == STAIR_TOP_ID:
            all_segs = [("upper", seg_to_node)]
            total    = dist_to_node + stair_dist
            if seg_bot_to_exit:
                all_segs.append(("ground", seg_bot_to_exit))
                total += dist_bot_to_exit
                results.append({**_base(space),
                                 "total_distance_m":  round(total, 2),
                                 "art57_distance_m":  round(dist_to_node, 2),
                                 "art61_distance_m":  round(total-dist_to_node, 2),
                                 "segments": all_segs,
                                 "reachable": True})
            else:
                results.append(_unreachable(space))
            continue

        # Greedy walk from best node toward FireExit
        all_segs      = [("own_floor", seg_to_node)]
        total_dist    = dist_to_node
        current       = best_loc
        cur_elev      = elev
        visited       = {best_node["id"]}
        success       = False
        used_stair    = False

        for _ in range(25):
            fg_cur = grid_map._get_floor(cur_elev)
            if fg_cur is None: break

            # Try FireExit directly
            if abs(cur_elev - exit_elev) < 0.5:
                seg_e = fg_cur.astar(current, _snap(exit_loc, fg_cur))
                if seg_e:
                    all_segs.append(("ground", seg_e))
                    total_dist += fg_cur.path_distance(seg_e)
                    success = True; break

            # Try stair on upper floor
            if (not used_stair and snapped_top is not None
                    and abs(cur_elev - exit_elev) > 0.5):
                seg_t = fg_cur.astar(current, snapped_top)
                if seg_t:
                    all_segs.append(("upper", seg_t))
                    total_dist += fg_cur.path_distance(seg_t) + stair_dist
                    used_stair = True
                    current    = snapped_bot
                    cur_elev   = exit_elev
                    if seg_bot_to_exit:
                        all_segs.append(("ground", seg_bot_to_exit))
                        total_dist += dist_bot_to_exit
                        success = True
                    break

            # Greedy: next door closest to exit
            candidates = [
                d for d in all_doors_plus
                if d["id"] not in visited and d.get("location")
                and abs(d.get("storey_elevation_m",
                        d["location"][2] if len(d["location"]) > 2 else 0.0)
                        - cur_elev) < 1.0
            ]
            candidates.sort(
                key=lambda d: _dist((d["location"][0], d["location"][1]), exit_loc))

            moved = False
            for nd in candidates:
                nd_loc = _snap((nd["location"][0], nd["location"][1]), fg_cur)
                seg_nd = fg_cur.astar(current, nd_loc)
                if not seg_nd: continue
                if nd["id"] == STAIR_TOP_ID and not used_stair:
                    all_segs.append(("upper", seg_nd))
                    total_dist += fg_cur.path_distance(seg_nd) + stair_dist
                    visited.add(nd["id"]); used_stair = True
                    current = snapped_bot; cur_elev = exit_elev
                    if seg_bot_to_exit:
                        all_segs.append(("ground", seg_bot_to_exit))
                        total_dist += dist_bot_to_exit; success = True
                    moved = True; break
                else:
                    all_segs.append(("own_floor", seg_nd))
                    total_dist += fg_cur.path_distance(seg_nd)
                    visited.add(nd["id"]); current = nd_loc
                    moved = True; break

            if success or not moved: break

        if not success:
            results.append(_unreachable(space)); continue

        results.append({
            **_base(space),
            "total_distance_m": round(total_dist, 2),
            "art57_distance_m": round(dist_to_node, 2),
            "art61_distance_m": round(total_dist - dist_to_node, 2),
            "segments":         all_segs,
            "reachable":        True,
        })

    reachable = sum(1 for r in results if r["reachable"])
    print(f"[PathFinder] Done. {reachable}/{len(results)} spaces reachable.")
    for r in results:
        if not r["reachable"]:
            print(f"  UNREACHABLE: {r['space_name']}  storey={r['storey_elevation_m']}m")
    return results


# ── Result helpers ────────────────────────────────────────────────────────────

def _base(space):
    return {
        "space_id":           space["id"],
        "space_name":         space.get("name", ""),
        "storey_id":          space.get("storey_id"),
        "storey_elevation_m": space.get("storey_elevation_m", 0.0),
        "has_two_exits":      False,
    }


def _unreachable(space):
    return {
        **_base(space),
        "total_distance_m": float("inf"),
        "art57_distance_m": 0.0,
        "art61_distance_m": 0.0,
        "segments":         [],
        "reachable":        False,
    }
