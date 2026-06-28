"""
main.py
SCIE Fire Evacuation Travel Distance Compliance Checker
RT-SCIE | Portaria 1532/2008 + 135/2020

Usage:
    python main.py              ← auto-detects .ifc file in project folder
    python main.py model.ifc    ← specify path manually
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ifc_parser
import ids_validator
import path_finder
import rule_checker
import bcf_reporter
import visualizer
from wall_aware_paths import GridMap

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IDS_PATH   = os.path.join(BASE_DIR, "ids",   "scie_ids.json")
RULES_PATH = os.path.join(BASE_DIR, "rules", "scie_rules.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def run(ifc_path: str):
    print("\n" + "=" * 60)
    print("SCIE FIRE EVACUATION COMPLIANCE CHECKER")
    print("RT-SCIE | Portaria 1532/2008 + 135/2020")
    print("=" * 60)

    # ── Step 1: Load IFC ──────────────────────────────────────────
    print("\n[Step 1] Loading IFC model...")
    model = ifc_parser.load_ifc(ifc_path)

    # ── Step 2: IDS Validation ────────────────────────────────────
    print("\n[Step 2] Running IDS validation...")
    ids_result = ids_validator.validate(model, IDS_PATH)

    if not ids_result["passed"]:
        print("\n[STOP] IDS validation FAILED.")
        print("Fix the following issues in your IFC model:")
        for e in ids_result["errors"]:
            print(f"  [{e['check_id']}] {e['message']}")
        return None

    # ── Step 3: Extract IFC data ──────────────────────────────────
    print("\n[Step 3] Extracting spaces, doors and relationships...")
    rules    = rule_checker.load_rules(RULES_PATH)
    ifc_data = ifc_parser.extract_all(model, rules)

    if len(ifc_data["spaces"]) == 0:
        print("[STOP] No spaces found.")
        return None

    if len(ifc_data["space_boundaries"]) == 0:
        print("[WARNING] No space boundaries found.")

    # ── Step 4: Build wall-aware grid map ─────────────────────────
    print("\n[Step 4] Building wall-aware grid map (one per floor)...")
    try:
        grid_map = GridMap(ifc_data)
    except Exception as e:
        grid_map = None
        print(f"[WARNING] Could not build grid map: {e}")

    # ── Step 5: Calculate evacuation paths ───────────────────────
    print("\n[Step 5] Calculating evacuation paths (direct A*)...")
    path_results = path_finder.calculate_all_paths(grid_map, ifc_data)

    if not path_results:
        print("[STOP] No paths calculated.")
        return None

    # ── Step 6: Check SCIE rules ──────────────────────────────────
    building_ut = ifc_data.get("building_ut", "UT_I")
    print(f"\n[Step 6] Checking SCIE rules for {building_ut}...")
    check_results = rule_checker.check_all(path_results, RULES_PATH, building_ut)

    # ── Step 7: BCF report ────────────────────────────────────────
    print("\n[Step 7] Generating BCF report...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bcf_reporter.generate_bcf_issues(check_results, OUTPUT_DIR)

    # ── Step 8: Floor plan visualisation ─────────────────────────
    print("\n[Step 8] Generating floor plan PNG...")
    violations = check_results["violation_list"]
    storeys    = ifc_data["storeys"]

    if storeys:
        for storey in storeys:
            has_spaces = any(
                s["storey_id"] == storey["id"]
                for s in ifc_data["spaces"]
            )
            if not has_spaces:
                continue
            name     = storey["name"]
            svg_path = os.path.join(OUTPUT_DIR, f"floor_{name}.svg")
            visualizer.generate_svg(
                path_results, violations, ifc_data,
                svg_path,
                storey_elevation=storey["elevation_m"],
                grid_map=grid_map,
            )
    else:
        svg_path = os.path.join(OUTPUT_DIR, "floor_plan.svg")
        visualizer.generate_svg(
            path_results, violations, ifc_data,
            svg_path, grid_map=grid_map,
        )

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  Total spaces checked : {check_results['total_spaces']}")
    print(f"  Compliant            : {check_results['compliant']}")
    print(f"  SCIE violations      : {check_results['violations']}")

    if check_results["violations"] == 0:
        print("\n  ✅  PASS — All spaces comply with SCIE Art.57 requirements.")
    else:
        print(f"\n  ❌  FAIL — {check_results['violations']} violation(s) found.")
        print(f"  See: {OUTPUT_DIR}/")

    print("=" * 60)
    return check_results


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        ifc_path = sys.argv[1]
    else:
        ifc_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".ifc")]
        if not ifc_files:
            print("[ERROR] No .ifc file found in the project folder.")
            sys.exit(1)
        if len(ifc_files) > 1:
            print(f"[INFO] Multiple IFC files found: {ifc_files}")
            print(f"[INFO] Using: {ifc_files[0]}")
        ifc_path = os.path.join(BASE_DIR, ifc_files[0])
        print(f"[INFO] Auto-detected IFC file: {ifc_files[0]}")

    run(ifc_path)
