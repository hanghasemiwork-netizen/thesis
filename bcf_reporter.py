"""
bcf_reporter.py
Generates BCF (BIM Collaboration Format) issues for each violation.
Outputs a simple BCF-compatible JSON and a human-readable report.
"""

import json
import os
from datetime import datetime


def generate_bcf_issues(check_results: dict, output_dir: str) -> str:
    """
    Generate BCF issues from check results.
    Creates:
      - violations_report.json  (machine-readable)
      - violations_report.txt   (human-readable)

    Returns path to the JSON report.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    violations = check_results["violation_list"]

    # ── BCF-style JSON ────────────────────────────────────────────────────────
    bcf_topics = []
    for i, v in enumerate(violations):
        topic = {
            "guid": f"SCIE-{i+1:04d}",
            "type": "Issue",
            "status": "Open",
            
            "title": f"Violation — {v['space_name']}",
            "description": (
                f"Rule: {v['rule_id']}\n"
                
                f"{v['description']}\n\n"
                f"Space: {v['space_name']} (ID: {v['space_id']})\n"
                f"Local risk: {v.get('local_risk', 'N/A')}\n"
                f"Path type: {v.get('path_type', 'N/A')}\n\n"
                f"Measured distance: {v['measured_m']} m\n"
                f"Maximum allowed:   {v['limit_m']} m\n"
                f"Excess:            +{v['excess_m']} m"
            ),
            "creation_date": timestamp,
            "ifc_guids": [v["space_id"]],
            "labels": [v["rule_id"]]
        }
        bcf_topics.append(topic)

    bcf_output = {
        "metadata": {
            "generated": timestamp,
            "regulation": "RT-SCIE (Portaria 1532/2008 + 135/2020)",
            "total_spaces_checked": check_results["total_spaces"],
            "compliant": check_results["compliant"],
            "violations_found": check_results["violations"]
        },
        "topics": bcf_topics
    }

    json_path = os.path.join(output_dir, "violations_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bcf_output, f, indent=2)

    # ── Human-readable TXT ────────────────────────────────────────────────────
    txt_path = os.path.join(output_dir, "violations_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("SCIE FIRE EVACUATION COMPLIANCE REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated:  {timestamp}\n")
        f.write(f"Regulation: RT-SCIE (Portaria 1532/2008 + 135/2020)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total spaces checked: {check_results['total_spaces']}\n")
        f.write(f"Compliant:            {check_results['compliant']}\n")
        f.write(f"Violations found:     {check_results['violations']}\n")
        f.write("=" * 60 + "\n\n")

        if not violations:
            f.write("✓ All spaces comply with SCIE evacuation distance requirements.\n")
        else:
            for i, v in enumerate(violations):
                f.write(f"ISSUE #{i+1}\n")
                f.write(f"  Rule:     {v['rule_id']}\n")
                
                f.write(f"  Space:    {v['space_name']}\n")
                f.write(f"  Space ID: {v['space_id']}\n")
                f.write(f"  Risk:     Local risk {v.get('local_risk', 'N/A')}\n")
                f.write(f"  Path:     {v.get('path_type', 'N/A')}\n")
                f.write(f"  Measured: {v['measured_m']} m\n")
                f.write(f"  Limit:    {v['limit_m']} m\n")
                f.write(f"  Excess:   +{v['excess_m']} m\n")
                f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("Summary by rule:\n")
        for rid, info in check_results["summary_by_rule"].items():
            f.write(f"  {rid} ({info['article']}): "
                    f"{info['count']} violation(s)\n")

    print(f"\n[BCF] Report saved:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")

    return json_path
