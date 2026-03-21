import ast
import re
from collections import defaultdict
from typing import Dict, List, Tuple


def _collect_assigned_names(node: ast.AST, scope: Dict[str, int]) -> List[str]:
    assigned = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    else:
        targets = []
    if targets:
        for target in targets:
            if isinstance(target, ast.Name):
                assigned.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List, ast.Starred)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        assigned.append(elt.id)
            elif isinstance(target, ast.Attribute):
                continue
    elif isinstance(node, ast.For):
        if isinstance(node.target, ast.Name):
            assigned.append(node.target.id)
    elif isinstance(node, ast.With):
        if isinstance(node.items[0].optional_vars, ast.Name):
            assigned.append(node.items[0].optional_vars.id)
    elif isinstance(node, ast.AsyncWith):
        if isinstance(node.items[0].optional_vars, ast.Name):
            assigned.append(node.items[0].optional_vars.id)
    elif isinstance(node, ast.comprehension):
        if isinstance(node.target, ast.Name):
            assigned.append(node.target.id)
    for child in ast.iter_child_nodes(node):
        assigned.extend(_collect_assigned_names(child, scope))
    return assigned


def _detect_shadowing(
    func_code: str,
) -> Tuple[Dict[str, List[str]], List[Tuple[int, str, str]]]:
    tree = ast.parse(func_code)
    func_node = tree.body[0] if isinstance(tree, ast.Module) and tree.body else None
    if not func_node or not isinstance(func_node, ast.FunctionDef):
        raise ValueError(
            "Provided code does not contain a top-level function definition."
        )

    scopes: List[Dict[str, int]] = [{}]
    shadow_map: Dict[str, List[str]] = defaultdict(list)
    shadow_nodes: List[Tuple[int, str, str]] = []

    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", 0)
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.With,
                ast.AsyncWith,
                ast.ListComp,
                ast.DictComp,
                ast.SetComp,
                ast.GeneratorExp,
            ),
        ):
            scopes.append({})

        assigned = _collect_assigned_names(node, scopes[-1])
        for name in assigned:
            for outer_scope in reversed(scopes[:-1]):
                if name in outer_scope:
                    shadow_map[name].append(name)
                    suffix = outer_scope[name]
                    outer_scope[name] += 1
                    new_name = f"{name}_{suffix}"
                    shadow_nodes.append((lineno, name, new_name))
                    scopes[-1][new_name] = 0
                    break
            else:
                scopes[-1][name] = 0

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.With,
                ast.AsyncWith,
                ast.ListComp,
                ast.DictComp,
                ast.SetComp,
                ast.GeneratorExp,
            ),
        ):
            scopes.pop()

    return dict(shadow_map), shadow_nodes


def refactor_variable_shadowing(source: str) -> str:
    """Detect and rename variables that shadow outer‑scope names inside a function.
    Works on a *string* containing Python source and returns transformed code with
    inner‑scope variables renamed using a numeric suffix (e.g., ``x`` → ``x_1``).
    """
    lines = source.splitlines()
    # Locate top‑level function definitions
    func_ranges = []
    start = None
    for i, line in enumerate(lines):
        if re.search(r"\bdef\s+\w+\s*\(", line):
            start = i
        if (
            start is not None
            and line.strip()
            and not line.startswith(" ")
            and not line.startswith("\t")
        ):
            if start < i:
                func_ranges.append((start, i))
            start = None
    if start is not None:
        func_ranges.append((start, len(lines)))

    new_lines = lines[:]
    offset = 0
    for func_start, func_end in func_ranges:
        func_source = "\n".join(new_lines[func_start:func_end])
        try:
            _, shadow_nodes = _detect_shadowing(func_source)
        except Exception:
            continue
        # Apply renamings from bottom to top
        for lineno, old_name, new_name in sorted(shadow_nodes, key=lambda x: -x[0]):
            abs_lineno = lineno + offset
            if abs_lineno < len(new_lines):
                new_lines[abs_lineno] = new_lines[abs_lineno].replace(
                    old_name, new_name
                )
        offset += func_end - func_start
    return "\n".join(new_lines)
