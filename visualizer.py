"""
visualizer.py
Generates 2D floor plan PNGs showing evacuation paths.

Changes from previous version:
- Path drawing uses seg_upper / seg_ground from path_finder (direct A* coords)
  instead of graph node sequences.
- Each floor PNG draws only spaces on that floor.
- Ground floor PNG also shows upper floor ground segments (purple dashed).
- No elevation filter on path coordinates — full path drawn end to end.
- Farthest point logic removed from here (handled in path_finder).
"""

import os
import numpy as np


def generate_svg(path_results: list, violations: list,
                 ifc_data: dict, output_path: str,
                 storey_elevation: float = None,
                 grid_map=None):
    """
    Generate 2D floor plan PNG.

    Args:
        path_results : output of path_finder.calculate_all_paths()
        violations   : output of rule_checker.check_all()["violation_list"]
        ifc_data     : full ifc_data dict (for spaces, doors, stairs)
        output_path  : output file path (.svg or .png)
        storey_elevation : elevation of the floor to render
        grid_map     : GridMap instance for A* wall rendering
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.lines import Line2D
    except ImportError:
        print("[VIZ] matplotlib not installed.")
        return

    spaces  = ifc_data["spaces"]
    doors   = ifc_data["doors"]
    stairs  = ifc_data.get("stairs", [])

    stair_data = stairs[0] if stairs else None
    bottom_loc = (stair_data["bottom_loc"][0], stair_data["bottom_loc"][1]) if stair_data else None
    top_loc    = (stair_data["top_loc"][0],    stair_data["top_loc"][1])    if stair_data else None

    violated_space_ids = {v["space_id"] for v in violations}
    is_gnd = storey_elevation is not None and abs(storey_elevation) < 0.5

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#F0F4F8")
    fig.patch.set_facecolor("#FFFFFF")

    has_content = False

    # ── Space polygons ────────────────────────────────────────────────────────
    for space in spaces:
        if storey_elevation is not None:
            if abs(space.get("storey_elevation_m", 0.0) - storey_elevation) > 0.5:
                continue

        fp  = space.get("footprint", [])
        loc = space["location"]
        sid = space["id"]
        violated = sid in violated_space_ids

        fill = "#FFCCCC" if violated else "#D6E8F5"
        edge = "#C0392B" if violated else "#2471A3"

        if fp and len(fp) > 2:
            pts = np.array(fp)
            try:
                ax.add_patch(MplPolygon(pts, closed=True,
                                        facecolor=fill, edgecolor=edge,
                                        linewidth=2.0, alpha=0.85, zorder=2))
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
            except Exception:
                cx, cy = loc[0], loc[1]
        else:
            cx, cy = loc[0], loc[1]
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx - 1.5, cy - 1.0), 3.0, 2.0,
                boxstyle="round,pad=0.1",
                facecolor=fill, edgecolor=edge,
                linewidth=2.0, alpha=0.85, zorder=2))

        has_content = True
        ax.text(cx, cy + 0.25, space.get("name", ""),
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="#1A1A2E", zorder=5)

        # distance label
        r = next((x for x in path_results if x["space_id"] == sid), None)
        if r:
            if r["reachable"]:
                dist_text  = f"{r['total_distance_m']}m"
                dist_color = "#C0392B" if violated else "#1D6A3A"
            else:
                dist_text  = "NO PATH"
                dist_color = "#C0392B"
            ax.text(cx, cy - 0.35, dist_text,
                    ha="center", va="center",
                    fontsize=6, color=dist_color,
                    fontweight="bold", zorder=5)

    # ── Evacuation paths ──────────────────────────────────────────────────────
    if grid_map is not None:
        print("[VIZ] Using global wall-aware grid map for rendering.")

    drawn_ground_segs = set()

    for r in path_results:
        if not r["reachable"]:
            continue

        space_elev   = r["storey_elevation_m"]
        space_is_gnd = abs(space_elev - (storey_elevation or 0.0)) < 0.5
        violated     = r["space_id"] in violated_space_ids
        segments     = r.get("segments", [])

        if storey_elevation is None:
            continue

        for seg_type, seg in segments:
            if not seg or len(seg) < 2:
                continue

            if seg_type in ("own_floor", "upper"):
                # Draw on the space's own floor PNG only
                if not space_is_gnd:
                    continue
                if not (abs(space_elev - storey_elevation) < 0.5):
                    continue
                col = "#C0392B" if violated else "#1D9E75"
                _draw_seg(ax, seg, col, 2.2, "-")

            elif seg_type == "ground":
                if is_gnd:
                    # Ground floor spaces — draw in green on ground floor PNG
                    if abs(space_elev - storey_elevation) < 0.5:
                        col = "#C0392B" if violated else "#1D9E75"
                        _draw_seg(ax, seg, col, 2.2, "-")
                    else:
                        # Upper floor space — show ground segment in purple dashed
                        key = id(seg)
                        if key not in drawn_ground_segs:
                            drawn_ground_segs.add(key)
                            _draw_seg(ax, seg, "#7B2D8B", 1.5, "--")

    # ── Door / virtual passage markers — oriented yellow rectangles ───────────
    import matplotlib.transforms as mtransforms

    for door in doors:
        dloc = door.get("location")
        if not dloc:
            continue
        d_elev = door.get("storey_elevation_m",
                          dloc[2] if len(dloc) > 2 else 0.0)
        if storey_elevation is not None and abs(d_elev - storey_elevation) > 1.2:
            continue

        sx, sy  = dloc[0], dloc[1]
        w       = door.get("width_m", 0.915)
        depth   = 0.18   # visual wall thickness
        angle   = door.get("angle_deg", 0.0)

        if door.get("is_fire_exit"):
            face_col = "#ABEBC6"; edge_col = "#1D9E75"; lw = 2.0
        elif door.get("is_virtual"):
            face_col = "#FCF3CF"; edge_col = "#B7950B"; lw = 1.5
        else:
            face_col = "#FEF9E7"; edge_col = "#E67E22"; lw = 1.2

        rect = plt.Rectangle(
            (-w / 2, -depth / 2), w, depth,
            facecolor=face_col, edgecolor=edge_col,
            linewidth=lw, alpha=0.55, zorder=6
        )
        t = (mtransforms.Affine2D()
             .rotate_deg(angle)
             .translate(sx, sy)
             + ax.transData)
        rect.set_transform(t)
        ax.add_patch(rect)

        # Label FireExit
        if door.get("is_fire_exit"):
            ax.text(sx, sy + 0.6, "EXIT",
                    ha="center", fontsize=6, fontweight="bold",
                    color="#1D9E75", zorder=7)

    # ── Stair landing — rectangle from bounds ─────────────────────────────────
    for stair in ifc_data.get("stairs", []):
        bounds = stair.get("bounds")
        if not bounds:
            continue
        mn_x, mn_y, mx_x, mx_y = bounds
        # Show on the floor where the landing is
        for loc_key, label in [("bottom_loc", "▼"), ("top_loc", "▲")]:
            sloc = stair.get(loc_key)
            if not sloc:
                continue
            s_elev = sloc[2]
            if storey_elevation is not None and abs(s_elev - storey_elevation) > 1.2:
                continue
            ax.add_patch(plt.Rectangle(
                (mn_x, mn_y), mx_x - mn_x, mx_y - mn_y,
                facecolor="#FAD7A0", edgecolor="#E67E22",
                linewidth=1.5, alpha=0.45, zorder=5
            ))
            ax.text((mn_x + mx_x) / 2, (mn_y + mx_y) / 2,
                    f"{label} STAIR",
                    ha="center", va="center", fontsize=5,
                    color="#784212", fontweight="bold", zorder=7)

    # ── Axis, title, legend ───────────────────────────────────────────────────
    if not has_content:
        print(f"[VIZ] No content for storey {storey_elevation}m — skipping")
        plt.close()
        return

    ax.autoscale_view()
    ax.set_aspect("equal")
    margin = 2.0
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
    ax.set_ylim(ylim[0] - margin, ylim[1] + margin)

    ax.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC", alpha=0.7, zorder=1)
    ax.set_xlabel("X (meters)", fontsize=10)
    ax.set_ylabel("Y (meters)", fontsize=10)

    storey_label = (f"Floor {round(storey_elevation, 1)}m"
                    if storey_elevation is not None else "All floors")
    ax.set_title(f"SCIE Fire Evacuation Compliance — {storey_label}",
                 fontsize=13, fontweight="bold", pad=12)

    legend_elements = [
        mpatches.Patch(facecolor="#D6E8F5", edgecolor="#2471A3", label="Space — compliant"),
        mpatches.Patch(facecolor="#FFCCCC", edgecolor="#C0392B", label="Space — violated"),
        Line2D([0],[0], color="#1D9E75", lw=2,   label="Compliant path"),
        Line2D([0],[0], color="#C0392B", lw=2,   label="Violated path"),
        Line2D([0],[0], color="#7B2D8B", lw=1.5, linestyle="--",
               label="Upper floor → exit (ground floor)"),
        mpatches.Patch(facecolor="#FEF9E7", edgecolor="#E67E22", alpha=0.55, label="Door"),
        mpatches.Patch(facecolor="#ABEBC6", edgecolor="#1D9E75", alpha=0.55, label="Fire exit door"),
        mpatches.Patch(facecolor="#FCF3CF", edgecolor="#B7950B", alpha=0.55, label="Virtual passage"),
        mpatches.Patch(facecolor="#FAD7A0", edgecolor="#E67E22", alpha=0.45, label="Stair landing"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")

    plt.tight_layout()

    png_path = output_path.replace(".svg", ".png")
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[VIZ] Saved: {png_path}")


def _draw_seg(ax, seg, color, lw, ls):
    """Draw a path segment with an arrow at the end."""
    xs = [p[0] for p in seg]
    ys = [p[1] for p in seg]
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls,
            alpha=0.85, zorder=10, solid_capstyle="round")
    if len(xs) > 1:
        ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                    zorder=11)
