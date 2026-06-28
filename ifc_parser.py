"""
ifc_parser.py
Extracts spaces, doors, stairs, columns and relationships from an IFC model.

Unit handling:
  - get_length_unit_scale() is called ONCE in extract_all().
  - The resulting scale factor is stored in ifc_data["scale"] and passed
    to every function that needs it. No function computes scale internally.
  - ifcopenshell.geom with USE_WORLD_COORDS=True returns geometry already
    in metres for most exporters. The heuristic check
    (max coordinate > 100) detects millimetre exports and applies scale.

Columns:
  - IfcColumn footprints are extracted and stored in ifc_data["columns"].
  - wall_aware_paths.py uses them as additional grid obstacles.
"""

import os
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement


# ── Settings ─────────────────────────────────────────────────────────────────

def _make_settings():
    s = ifcopenshell.geom.settings()
    s.set(s.USE_WORLD_COORDS, True)
    return s


def load_ifc(ifc_path: str):
    if not os.path.exists(ifc_path):
        raise FileNotFoundError(f"IFC file not found: {ifc_path}")
    model = ifcopenshell.open(ifc_path)
    print(f"[IFC] Loaded: {ifc_path}")
    print(f"[IFC] Schema: {model.schema}")
    return model


# ── Unit detection ────────────────────────────────────────────────────────────

def get_length_unit_scale(model) -> float:
    """
    Returns the scale factor to convert IFC numerical values to metres.
    Called ONCE in extract_all(). Result stored in ifc_data["scale"].
    """
    try:
        for ua in model.by_type("IfcUnitAssignment"):
            for unit in ua.Units:
                if not hasattr(unit, "UnitType"):
                    continue
                if unit.UnitType != "LENGTHUNIT":
                    continue
                prefix = getattr(unit, "Prefix", None)
                name   = getattr(unit, "Name",   None)
                if name == "METRE":
                    if prefix == "MILLI":  return 0.001
                    if prefix == "CENTI":  return 0.01
                    return 1.0
                if name == "FOOT":  return 0.3048
                if name == "INCH":  return 0.0254
    except Exception:
        pass
    return 0.001   # safe default — Revit exports mm


def _apply_scale_if_needed(pts: list, scale: float) -> list:
    """Apply scale only when coordinates appear to still be in mm (>100m)."""
    if not pts or scale == 1.0:
        return pts
    if max(abs(p[0]) for p in pts) > 100:
        return [(p[0] * scale, p[1] * scale) for p in pts]
    return pts


# ── Building  ──────────────────────────────────────────────────────
def get_building_ut(model) -> str:
    buildings = model.by_type("IfcBuilding")
    if not buildings:
        print("[WARNING] No IfcBuilding found. Defaulting to UT_I.")
        return "UT_I"
    building = buildings[0]

    try:
        for rel in getattr(building, "HasAssociations", []):
            if rel.is_a("IfcRelAssociatesClassification"):
                cls = rel.RelatingClassification
                if cls.is_a("IfcClassificationReference"):
                    code = getattr(cls, "Identification", getattr(cls, "ItemReference", ""))
                    sys_src = getattr(cls, "ReferencedSource", None)
                    sys_name = getattr(sys_src, "Name", "") if sys_src else getattr(cls, "Name", "")
                    
                    code_str = str(code).upper()
                    sys_str = str(sys_name).upper()
                    name_str = str(getattr(cls, "Name", "")).upper()

                    if "RT-SCIE" in sys_str or "UT_" in code_str or "UT_" in name_str:
                        if "UT_" in code_str: return code_str
                        if "UT_" in name_str: return name_str
    except Exception:
        pass

    try:
        for rel in getattr(building, "IsDefinedBy", []):
            if rel.is_a("IfcRelDefinesByProperties"):
                pset = rel.RelatingPropertyDefinition
                if pset.is_a("IfcPropertySet") and pset.Name == "Pset_BuildingCommon":
                    for prop in getattr(pset, "HasProperties", []):
                        if prop.Name == "OccupancyType":
                            val = getattr(prop, "NominalValue", None)
                            if val and val.wrappedValue:
                                return str(val.wrappedValue).upper()
    except Exception:
        pass

    return "UT_I"

# ── IFC property helpers ──────────────────────────────────────────────────────

def get_property(element, pset_name: str, prop_name: str):
    try:
        for rel in element.IsDefinedBy:
            if rel.is_a("IfcRelDefinesByProperties"):
                pset = rel.RelatingPropertyDefinition
                if pset.is_a("IfcPropertySet"):
                    if pset_name == "any" or pset.Name == pset_name:
                        for prop in pset.HasProperties:
                            if prop.Name == prop_name:
                                if hasattr(prop, "NominalValue") and prop.NominalValue:
                                    return prop.NominalValue.wrappedValue
    except Exception:
        pass
    return None


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _apply_transform(matrix, x: float, y: float) -> tuple:
    wx = matrix[0][0]*x + matrix[0][1]*y + matrix[0][3]
    wy = matrix[1][0]*x + matrix[1][1]*y + matrix[1][3]
    return (wx, wy)


def _interpolate_arc(p1, pmid, p2, n=8) -> list:
    ax, ay = p1; bx, by = pmid; cx, cy = p2
    D = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(D) < 1e-10:
        return [p1, p2]
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    r  = np.sqrt((ax-ux)**2 + (ay-uy)**2)
    a0 = np.arctan2(ay-uy, ax-ux)
    am = np.arctan2(by-uy, bx-ux)
    a1 = np.arctan2(cy-uy, cx-ux)
    ccw_0m = (am-a0) % (2*np.pi)
    ccw_01 = (a1-a0) % (2*np.pi)
    if ccw_0m < ccw_01:
        angles = np.linspace(a0, a0+ccw_01, n+1)
    else:
        angles = np.linspace(a0, a0-(2*np.pi-ccw_01), n+1)
    return [(ux+r*np.cos(a), uy+r*np.sin(a)) for a in angles]


def _read_indexed_poly_curve(curve, matrix) -> list:
    if not curve.is_a("IfcIndexedPolyCurve"):
        return []
    coord_list = curve.Points.CoordList
    def _pt(idx):
        x, y = coord_list[idx-1]
        return _apply_transform(matrix, float(x), float(y))
    segments = curve.Segments
    if not segments:
        return [_pt(i+1) for i in range(len(coord_list))]
    pts = []
    for seg in segments:
        indices = list(seg.wrappedValue)
        if seg.is_a("IfcLineIndex"):
            for idx in indices[:-1]:
                pts.append(_pt(idx))
        elif seg.is_a("IfcArcIndex") and len(indices) >= 3:
            arc_pts = _interpolate_arc(_pt(indices[0]), _pt(indices[1]), _pt(indices[2]))
            pts.extend(arc_pts[:-1])
    return pts


def _read_trimmed_curve(curve, matrix, n=8) -> list:
    try:
        basis = curve.BasisCurve
        trim1 = curve.Trim1
        trim2 = curve.Trim2
        same  = curve.SenseAgreement
        if basis.is_a("IfcCircle"):
            centre = basis.Position
            loc    = centre.Location.Coordinates if hasattr(centre, "Location") else None
            cx_w   = _apply_transform(matrix, float(loc[0]), float(loc[1])) if loc else _apply_transform(matrix, 0, 0)
            cx_w, cy_w = cx_w[0], cx_w[1]
            scale  = np.sqrt(matrix[0][0]**2 + matrix[1][0]**2)
            r      = float(basis.Radius) * scale
            def _ang(ts):
                for t in ts:
                    try:
                        if hasattr(t, "wrappedValue"):
                            return np.radians(float(t.wrappedValue))
                        if hasattr(t, "Coordinates"):
                            px, py = float(t.Coordinates[0]), float(t.Coordinates[1])
                            wx, wy = _apply_transform(matrix, px, py)
                            return np.arctan2(wy-cy_w, wx-cx_w)
                    except Exception:
                        continue
                return 0.0
            a0 = _ang(trim1); a1 = _ang(trim2)
            if not same: a0, a1 = a1, a0
            if a1 < a0: a1 += 2*np.pi
            return [(cx_w+r*np.cos(a), cy_w+r*np.sin(a))
                    for a in np.linspace(a0, a1, n+1)]
        elif basis.is_a("IfcLine"):
            pts = []
            for ts in [trim1, trim2]:
                for t in ts:
                    try:
                        if hasattr(t, "Coordinates"):
                            pts.append(_apply_transform(matrix, float(t.Coordinates[0]), float(t.Coordinates[1])))
                            break
                    except Exception:
                        continue
            return pts if len(pts) == 2 else []
    except Exception:
        pass
    return []


def _profile_to_points(profile, matrix) -> list:
    ptype = profile.is_a()
    if ptype in ("IfcArbitraryClosedProfileDef", "IfcArbitraryProfileDefWithVoids"):
        curve = profile.OuterCurve
        if curve.is_a("IfcIndexedPolyCurve"):
            return _read_indexed_poly_curve(curve, matrix)
        if curve.is_a("IfcPolyline"):
            return [_apply_transform(matrix, float(p.Coordinates[0]), float(p.Coordinates[1]))
                    for p in curve.Points]
        if curve.is_a("IfcCompositeCurve"):
            pts = []
            for seg in curve.Segments:
                sub  = seg.ParentCurve
                same = getattr(seg, "SameSense", True)
                if sub.is_a("IfcIndexedPolyCurve"):
                    sp = _read_indexed_poly_curve(sub, matrix)
                elif sub.is_a("IfcPolyline"):
                    sp = [_apply_transform(matrix, float(p.Coordinates[0]), float(p.Coordinates[1]))
                          for p in sub.Points]
                elif sub.is_a("IfcTrimmedCurve"):
                    sp = _read_trimmed_curve(sub, matrix)
                else:
                    continue
                pts.extend(sp if same else sp[::-1])
            return pts
        if curve.is_a("IfcTrimmedCurve"):
            return _read_trimmed_curve(curve, matrix)
    elif ptype == "IfcRectangleProfileDef":
        w, h = profile.XDim/2.0, profile.YDim/2.0
        return [_apply_transform(matrix, x, y) for x, y in [(-w,-h),(w,-h),(w,h),(-w,h)]]
    elif ptype == "IfcCircleProfileDef":
        r = profile.Radius
        return [_apply_transform(matrix, r*np.cos(a), r*np.sin(a))
                for a in np.linspace(0, 2*np.pi, 16, endpoint=False)]
    return []


# ── Space footprint ───────────────────────────────────────────────────────────

def get_space_footprint(space, settings, scale: float) -> list:
    """Extract 2D footprint polygon of an IfcSpace in world metres."""
    try:
        rep = space.Representation
        if rep:
            for sub_rep in rep.Representations:
                for item in sub_rep.Items:
                    solid = item
                    while solid.is_a("IfcBooleanClippingResult"):
                        solid = solid.FirstOperand
                    if not solid.is_a("IfcExtrudedAreaSolid"):
                        continue
                    profile  = solid.SweptArea
                    sm       = ifcopenshell.util.placement.get_local_placement(space.ObjectPlacement)
                    sp_pos   = solid.Position
                    if sp_pos:
                        wm = sm @ ifcopenshell.util.placement.get_axis2placement(sp_pos)
                    else:
                        wm = sm
                    pts = _profile_to_points(profile, wm)
                    pts = _apply_scale_if_needed(pts, scale)
                    if len(pts) >= 3:
                        return pts
    except Exception:
        pass
    try:
        shape  = ifcopenshell.geom.create_shape(settings, space)
        verts  = shape.geometry.verts
        pts_3d = [(verts[i], verts[i+1], verts[i+2]) for i in range(0, len(verts), 3)]
        if not pts_3d:
            return []
        z_vals  = sorted(set(round(p[2], 3) for p in pts_3d))
        floor_z = z_vals[0]
        floor_pts = list(set((p[0], p[1]) for p in pts_3d if abs(p[2]-floor_z) < 0.001))
        return floor_pts
    except Exception:
        return []


def get_space_centroid(footprint: list) -> tuple:
    if not footprint:
        return (0.0, 0.0)
    return (sum(p[0] for p in footprint)/len(footprint),
            sum(p[1] for p in footprint)/len(footprint))


def get_element_location_geometry(element, settings) -> tuple:
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = shape.geometry.verts
        if verts:
            xs = verts[0::3]; ys = verts[1::3]; zs = verts[2::3]
            return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
    except Exception:
        pass
    return (0.0, 0.0, 0.0)


def get_element_location_placement(element, scale: float) -> tuple:
    try:
        m = ifcopenshell.util.placement.get_local_placement(element.ObjectPlacement)
        return (m[0][3]*scale, m[1][3]*scale, m[2][3]*scale)
    except Exception:
        return (0.0, 0.0, 0.0)


def get_storey_elevation(storey, scale: float) -> float:
    try:
        if hasattr(storey, "Elevation") and storey.Elevation is not None:
            return float(storey.Elevation) * scale
    except Exception:
        pass
    return 0.0


# ── Stairs ────────────────────────────────────────────────────────────────────

def get_stairs(model, settings) -> list:
    stairs = []
    for stair in model.by_type("IfcStair"):
        sid   = stair.GlobalId
        sname = stair.Name or "Stair"
        bottom_loc = top_loc = bounds = None

        for flight in model.by_type("IfcStairFlight"):
            belongs = False
            for rel in model.by_type("IfcRelAggregates"):
                if rel.RelatingObject.GlobalId == sid:
                    if any(o.GlobalId == flight.GlobalId for o in rel.RelatedObjects):
                        belongs = True
                        break
            if not belongs and sname.split(":")[0] in (flight.Name or ""):
                belongs = True
            if not belongs:
                continue
            try:
                shape = ifcopenshell.geom.create_shape(settings, flight)
                verts = shape.geometry.verts
                xs = verts[0::3]; ys = verts[1::3]; zs = verts[2::3]
                min_z = min(zs); max_z = max(zs)
                b_pts = [(xs[i],ys[i]) for i in range(len(zs)) if abs(zs[i]-min_z)<0.15]
                t_pts = [(xs[i],ys[i]) for i in range(len(zs)) if abs(zs[i]-max_z)<0.15]
                if b_pts:
                    bottom_loc = (sum(p[0] for p in b_pts)/len(b_pts),
                                  sum(p[1] for p in b_pts)/len(b_pts), min_z)
                if t_pts:
                    top_loc    = (sum(p[0] for p in t_pts)/len(t_pts),
                                  sum(p[1] for p in t_pts)/len(t_pts), max_z)
                bounds = (min(xs), min(ys), max(xs), max(ys))
            except Exception:
                continue

        if bottom_loc is None:
            try:
                shape = ifcopenshell.geom.create_shape(settings, stair)
                verts = shape.geometry.verts
                xs = verts[0::3]; ys = verts[1::3]; zs = verts[2::3]
                cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
                bottom_loc = (cx, cy, min(zs))
                top_loc    = (cx, cy, max(zs))
                bounds = (min(xs), min(ys), max(xs), max(ys))
            except Exception:
                bottom_loc = (0.0, 0.0, 0.0)
                top_loc    = (0.0, 0.0, 3.0)
                bounds     = (-1.0, -1.0, 1.0, 1.0)

        stairs.append({
            "id": sid, "name": sname,
            "bottom_loc": bottom_loc,
            "top_loc":    top_loc,
            "location":   bottom_loc,
            "bounds":     bounds,
        })
        print(f"[IFC] Stair found: {sname}")
    return stairs


# ── Columns ───────────────────────────────────────────────────────────────────

def get_columns(model, settings, scale: float) -> list:
    """
    Extract IfcColumn footprint polygons as obstacle zones for path finding.
    Returns list of dicts with 'footprint' (list of (x,y) in metres)
    and 'storey_elevation_m'.
    """
    columns = []
    storey_map = {}
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        for el in rel.RelatedElements:
            if el.is_a("IfcColumn"):
                st = rel.RelatingStructure
                if st.is_a("IfcBuildingStorey"):
                    storey_map[el.GlobalId] = get_storey_elevation(st, scale)

    for col in model.by_type("IfcColumn"):
        try:
            shape = ifcopenshell.geom.create_shape(settings, col)
            verts = shape.geometry.verts
            pts_3d = [(verts[i], verts[i+1], verts[i+2]) for i in range(0, len(verts), 3)]
            if not pts_3d:
                continue
            z_vals   = sorted(set(round(p[2], 3) for p in pts_3d))
            floor_z  = z_vals[0]
            floor_pts = list(set((p[0], p[1]) for p in pts_3d if abs(p[2]-floor_z) < 0.05))
            if len(floor_pts) < 3:
                continue
            # convex hull for a clean polygon
            from shapely.geometry import MultiPoint
            hull = MultiPoint(floor_pts).convex_hull
            if hull.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            elev = storey_map.get(col.GlobalId, floor_z)
            columns.append({
                "id":                col.GlobalId,
                "name":              col.Name or "Column",
                "footprint":         list(hull.exterior.coords),
                "storey_elevation_m": elev,
            })
        except Exception:
            continue

    print(f"[IFC] Columns found: {len(columns)}")
    return columns


# ── Door helpers ──────────────────────────────────────────────────────────────

def is_fire_exit_door(door, model) -> bool:
    for prop_name in ["FireExit", "IsFireExit"]:
        val = get_property(door, "any", prop_name)
        if val is True or val == "True" or val == 1:
            return True
    return False


def is_apartment_entrance_door(door, rules: dict) -> bool:
    ptype = getattr(door, "PredefinedType", None)
    if ptype and "ENTRANCE" in str(ptype).upper():
        return True
    cfg  = (rules or {}).get("apartment_entrance", {})
    checks = cfg.get("properties", [
        {"pset": "any", "property": "ApartmentEntrance"},
        {"pset": "any", "property": "IsApartmentEntrance"},
    ])
    for check in checks:
        val = get_property(door, check["pset"], check["property"])
        if val is True or str(val).lower() in ("true", "1", "yes"):
            return True
    val = get_property(door, "Pset_DoorCommon", "Purpose")
    if val:
        kws = cfg.get("purpose_keywords", ["entrance", "main entrance", "entrada"])
        if any(kw in str(val).lower() for kw in kws):
            return True
    return False


# ── Wall openings as virtual passages ─────────────────────────────────────────

def get_wall_openings(model, settings, scale: float,
                      spaces: list = None) -> tuple:
    """
    Find unfilled wall openings that represent passages (no door element).
    Returns (virtual_door_list, space_boundary_pairs).

    Elevation rule: each opening is assigned ONLY to spaces whose
    storey_elevation_m matches the opening's own z coordinate (±1.0m).
    This prevents a ground-floor opening from being linked to upper-floor spaces.
    """
    filled_ids = {rel.RelatingOpeningElement.GlobalId
                  for rel in model.by_type("IfcRelFillsElement")}

    # Build space elevation lookup for filtering
    space_elev_map = {}
    if spaces:
        for sp in spaces:
            space_elev_map[sp["id"]] = sp.get("storey_elevation_m", 0.0)

    opening_doors = []
    opening_sb    = []

    for rel_void in model.by_type("IfcRelVoidsElement"):
        opening = rel_void.RelatedOpeningElement
        wall    = rel_void.RelatingBuildingElement
        if opening.GlobalId in filled_ids:
            continue
        if not wall.is_a("IfcWall"):
            continue
        ref     = get_property(opening, "Pset_OpeningElementCommon", "Reference")
        ref_str = str(ref).lower() if ref else ""
        if not ref:
            continue
        if "curtain" in ref_str:
            continue
        if "door" not in ref_str and "opening" not in ref_str:
            continue

        # Get actual 3D location and dimensions from geometry
        try:
            shape = ifcopenshell.geom.create_shape(settings, opening)
            verts = shape.geometry.verts
            xs = verts[0::3]; ys = verts[1::3]; zs = verts[2::3]
            cx  = sum(xs)/len(xs); cy = sum(ys)/len(ys)
            loc = (cx, cy, min(zs))
            # Bounding box in XY gives opening dimensions
            dx  = max(xs) - min(xs)
            dy  = max(ys) - min(ys)
            # Wall angle from wall placement
            try:
                import math
                wmat       = ifcopenshell.util.placement.get_local_placement(wall.ObjectPlacement)
                wall_angle = math.degrees(math.atan2(wmat[1][0], wmat[0][0]))
            except Exception:
                wall_angle = 0.0
            # Width = dimension ALONG the wall direction
            # Rectangle drawn along the wall = same angle as wall
            if abs(abs(wall_angle) - 90.0) < 10.0:
                # Wall is vertical → opening runs along Y → width = dy
                opening_width = dy
                opening_angle = 90.0
            else:
                # Wall is horizontal → opening runs along X → width = dx
                opening_width = dx
                opening_angle = 0.0
        except Exception:
            continue

        opening_z = loc[2]  # actual floor elevation of this opening

        # Find spaces that touch this wall AND are on the same floor
        # AND whose footprint is within 0.5m of the opening location
        from shapely.geometry import Point as _Point, Polygon as _Poly
        opening_pt = _Point(loc[0], loc[1])
        touching = []
        for sb in model.by_type("IfcRelSpaceBoundary"):
            if (sb.RelatedBuildingElement and
                    sb.RelatedBuildingElement.GlobalId == wall.GlobalId):
                sp_id = sb.RelatingSpace.GlobalId
                if sp_id in touching:
                    continue
                # Elevation check — only accept spaces on the same floor
                sp_elev = space_elev_map.get(sp_id)
                if sp_elev is not None and abs(sp_elev - opening_z) > 1.0:
                    continue
                # Proximity check — opening must be within 0.5m of space footprint
                if spaces:
                    sp_obj = next((s for s in spaces if s["id"] == sp_id), None)
                    if sp_obj and sp_obj.get("footprint"):
                        try:
                            sp_poly = _Poly(sp_obj["footprint"])
                            if not sp_poly.is_valid:
                                sp_poly = sp_poly.buffer(0)
                            if sp_poly.distance(opening_pt) > 0.5:
                                continue
                        except Exception:
                            pass
                touching.append(sp_id)

        if len(touching) < 2:
            continue

        opening_doors.append({
            "id":                 opening.GlobalId,
            "name":               f"WallOpening_{opening.GlobalId[:8]}",
            "location":           loc,
            "storey_elevation_m": opening_z,
            "width_m":            opening_width,
            "height_m":           2.2,
            "angle_deg":          opening_angle,
            "is_fire_exit":       False,
            "is_virtual":         True,
            "ifc_ref":            opening,
        })
        for sp_id in touching:
            opening_sb.append((sp_id, opening.GlobalId))

    return opening_doors, opening_sb


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_all(model, rules: dict) -> dict:
    """
    Extract all IFC data needed by the pipeline.
    Unit scale is detected ONCE here and applied everywhere.

    Returns ifc_data dict with keys:
        spaces, doors, storeys, stairs, columns,
        space_boundaries, scale
    """
    settings = _make_settings()

    # ── Detect unit scale ONCE ────────────────────────────────────
    scale = get_length_unit_scale(model)
    print(f"[IFC] Length unit scale: {scale} (1.0 = metres, 0.001 = mm)")
    
    building_ut = get_building_ut(model)
    print(f"[IFC] Detected Building Type: {building_ut}")
    # ── Map openings → doors ──────────────────────────────────────
    opening_to_door = {}
    for rel in model.by_type("IfcRelFillsElement"):
        try:
            op = rel.RelatingOpeningElement
            el = rel.RelatedBuildingElement
            if op and el and el.is_a("IfcDoor"):
                opening_to_door[op.GlobalId] = el.GlobalId
        except Exception:
            continue

    # ── Snap door z to nearest storey ────────────────────────────
    storey_elevs = []

    # ── Storeys ───────────────────────────────────────────────────
    storeys = []
    for st in model.by_type("IfcBuildingStorey"):
        elev = get_storey_elevation(st, scale)
        storey_elevs.append(elev)
        storeys.append({
            "id":           st.GlobalId,
            "name":         (st.Name or "Unknown").replace(" ", "_"),
            "elevation_m":  elev,
            "is_basement":  elev < 0,
            "is_high_floor": elev > 28.0,
            "ifc_ref":      st,
        })
    storey_elevs = sorted(set(round(e, 1) for e in storey_elevs))

    def snap_to_storey(z):
        if not storey_elevs:
            return 0.0
        cands = [e for e in storey_elevs if abs(e-z) <= 1.5]
        return min(cands, key=lambda e: abs(e-z)) if cands else storey_elevs[0]

    # ── Spaces ────────────────────────────────────────────────────
    spaces = []
    for sp in model.by_type("IfcSpace"):
        fp  = get_space_footprint(sp, settings, scale)
        cx, cy = get_space_centroid(fp)
        loc = (cx, cy, 0.0) if fp else get_element_location_placement(sp, scale)

        storey_id    = None
        storey_elev  = 0.0
        try:
            for rel in model.by_type("IfcRelAggregates"):
                if sp in rel.RelatedObjects and rel.RelatingObject.is_a("IfcBuildingStorey"):
                    storey_id   = rel.RelatingObject.GlobalId
                    storey_elev = get_storey_elevation(rel.RelatingObject, scale)
                    break
        except Exception:
            pass

        name      = sp.LongName or sp.Name or "Space"
        long_name = sp.LongName if hasattr(sp, "LongName") and sp.LongName else ""
        spaces.append({
            "id":                sp.GlobalId,
            "name":              name,
            "long_name":         long_name,
            "location":          loc,
            "footprint":         fp,
            "storey_id":         storey_id,
            "storey_elevation_m": storey_elev,
            "ifc_ref":           sp,
        })

    # ── Doors ─────────────────────────────────────────────────────
    doors = []
    for door in model.by_type("IfcDoor"):
        loc = get_element_location_geometry(door, settings)
        # Width and height in metres
        w = getattr(door, "OverallWidth",  None)
        h = getattr(door, "OverallHeight", None)
        w_m = float(w) * scale if w else 0.915  # default 915mm
        h_m = float(h) * scale if h else 2.100
        # Orientation angle from placement matrix (degrees, in XY plane)
        try:
            import math
            mat   = ifcopenshell.util.placement.get_local_placement(door.ObjectPlacement)
            angle = math.degrees(math.atan2(mat[1][0], mat[0][0]))
        except Exception:
            angle = 0.0
        doors.append({
            "id":                    door.GlobalId,
            "name":                  door.Name or "Door",
            "location":              loc,
            "storey_elevation_m":    snap_to_storey(loc[2] if len(loc) > 2 else 0.0),
            "width_m":               w_m,
            "height_m":              h_m,
            "angle_deg":             angle,
            "is_fire_exit":          is_fire_exit_door(door, model),
            "is_apartment_entrance": is_apartment_entrance_door(door, rules),
            "is_virtual":            False,
            "ifc_ref":               door,
        })

    # ── Stairs ────────────────────────────────────────────────────
    stairs = get_stairs(model, settings)

    # ── Columns ───────────────────────────────────────────────────
    columns = get_columns(model, settings, scale)

    # ── Space boundaries ──────────────────────────────────────────
    space_ids   = {s["id"]: s for s in spaces}
    door_ids    = {d["id"]: d for d in doors}
    space_boundaries = []

    for rel in model.by_type("IfcRelSpaceBoundary"):
        try:
            sp  = rel.RelatingSpace
            el  = rel.RelatedBuildingElement
            if not sp or not el:
                continue
            sp_obj = space_ids.get(sp.GlobalId)
            if not sp_obj:
                continue

            sp_elev = sp_obj.get("storey_elevation_m", 0.0)

            if el.is_a("IfcDoor"):
                d_obj = door_ids.get(el.GlobalId)
                if not d_obj:
                    continue
                # Elevation check — door must be on the same floor as space
                d_elev = d_obj.get("storey_elevation_m", 0.0)
                if abs(d_elev - sp_elev) > 1.0:
                    continue
                space_boundaries.append((sp.GlobalId, el.GlobalId))

            elif el.is_a("IfcOpeningElement"):
                d_id = opening_to_door.get(el.GlobalId)
                if not d_id:
                    continue
                d_obj = door_ids.get(d_id)
                if not d_obj:
                    continue
                d_elev = d_obj.get("storey_elevation_m", 0.0)
                if abs(d_elev - sp_elev) > 1.0:
                    continue
                space_boundaries.append((sp.GlobalId, d_id))

        except Exception:
            continue

    space_boundaries = list(set(space_boundaries))

    # ── Wall opening passages ─────────────────────────────────────
    # Pass spaces so get_wall_openings can filter by elevation
    wall_op_doors, wall_op_sb = get_wall_openings(model, settings, scale, spaces)
    doors.extend(wall_op_doors)
    space_boundaries.extend(wall_op_sb)
    space_boundaries = list(set(space_boundaries))
    if wall_op_doors:
        print(f"[IFC] {len(wall_op_doors)} wall opening(s) added as passage nodes")

    return {
        "spaces":           spaces,
        "doors":            doors,
        "storeys":          storeys,
        "stairs":           stairs,
        "columns":          columns,
        "space_boundaries": space_boundaries,
        "scale":            scale,
        "building_ut":      building_ut,
        # ... your other keys remain exactly the same
    }
