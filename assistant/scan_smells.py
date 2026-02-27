import re
import os

def scan_file_for_smells(filepath: str) -> list[str]:
    """Scan a Python file for common code smells using simple regex patterns.
    Returns a list of detected smell names."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return [f"Error reading file: {e}"]

    smells = []

    # 1. Mutable default argument (list, dict, or set)
    if re.search(r"def \w+\(.*?=\s*(\[\]|\{\}|set\(\))", content):
        smells.append("Mutable default argument")

    # 2. Bare except clause
    if re.search(r"^\s*except\s*:", content, re.MULTILINE):
        smells.append("Bare except clause")

    # 3. Using 'is' for value comparison (True, False, etc.)
    if re.search(r"\bis\s+True\b", content):
        smells.append("Using 'is' for True comparison")
    if re.search(r"\bis\s+False\b", content):
        smells.append("Using 'is' for False comparison")

    # 4. Inefficient string concatenation in a loop
    if re.search(r"\+=\s*str\(.*?\)", content):
        smells.append("Inefficient string concatenation in loop")

    # 5. Shadowing built‑in types (list, dict, str, etc.)
    if re.search(r"\blist\s*=", content):
        smells.append("Shadowing built‑in 'list'")
    if re.search(r"\bdict\s*=", content):
        smells.append("Shadowing built‑in 'dict'")
    if re.search(r"\bstr\s*=", content):
        smells.append("Shadowing built‑in 'str'")  

    # 6. Using 'is' for None check (not a smell but sometimes flagged)
    if re.search(r"\bis\s+None\b", content):
        # This is actually correct; we could ignore or note it
        pass

    return smells


def scan_project(root_path: str = "."):
    """Walk the project directory and report smells found in each Python file.
    Returns a dict mapping file paths to list of smells."""
    results = {}
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if fname.endswith(".py"):
                fpath = os.path.join(dirpath, fname)
                smells = scan_file_for_smells(fpath)
                if smells:
                    results[fpath] = smells
    return results


if __name__ == "__main__":
    # Quick demo when run directly
    import json
    report = scan_project()
    print(json.dumps(report, indent=2))
