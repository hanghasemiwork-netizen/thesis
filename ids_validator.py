"""
ids_validator.py
Validates IFC model completeness before compliance checking.
Reads rules from ids/scie_ids.json.
"""

import json
import os


def load_ids(ids_path: str) -> dict:
    with open(ids_path, "r") as f:
        return json.load(f)


def validate(model, ids_path: str) -> dict:
    """
    Validate the IFC model against IDS requirements.
    Returns:
        {
            "passed": True/False,
            "errors": [...],
            "warnings": [...]
        }
    """
    ids = load_ids(ids_path)
    errors = []
    warnings = []

    for check in ids["checks"]:
        cid = check["id"]
        entity = check["ifc_entity"]

        # Level 1 and 2: entity must exist
        if check["requirement"] == "at_least_one":
            try:
                items = model.by_type(entity)
                if len(items) == 0:
                    errors.append({
                        "check_id": cid,
                        "level": check["level"],
                        "message": check["error_message"]
                    })
                else:
                    print(f"[IDS] {cid} PASS — found {len(items)} {entity}")
            except Exception as e:
                errors.append({
                    "check_id": cid,
                    "level": check["level"],
                    "message": f"Error checking {entity}: {str(e)}"
                })

        # Level 3: at least one fire exit door
        elif check["requirement"] == "at_least_one_fire_exit":
            found = False
            try:
                doors = model.by_type("IfcDoor")
                for door in doors:
                    # Check explicit FireExit properties only
                    # (IsExternal / external boundary removed — false positives)
                    for prop_check in check["fire_exit_properties"]:
                        val = _get_prop(door, prop_check["pset"], prop_check["property"])
                        if val is True or val == "True" or val == 1:
                            found = True
                            break
                    if found:
                        break

            except Exception as e:
                pass

            if not found:
                # Hard error — pipeline cannot run without a fire exit
                errors.append({
                    "check_id": cid,
                    "level": 1,
                    "message": check["error_message"] +
                               " — pipeline stopped. Please mark at least one door "
                               "with FireExit = True in Revit before exporting to IFC."
                })
                print(f"[IDS] {cid} ERROR — no fire exit door found. Pipeline stopped.")
            else:
                print(f"[IDS] {cid} PASS — fire exit door found")

    # W-01 removed — local risk is inferred automatically from space names
    # via the keyword dictionary in scie_rules.json

    passed = len(errors) == 0

    print(f"\n[IDS] Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        print(f"[IDS] Errors ({len(errors)}):")
        for e in errors:
            print(f"  [{e['check_id']}] {e['message']}")
    if warnings:
        print(f"[IDS] Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  [{w['check_id']}] {w['message']}")

    return {
        "passed": passed,
        "errors": errors,
        "warnings": warnings
    }


def _get_prop(element, pset_name: str, prop_name: str):
    """Helper to get property value from element."""
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
