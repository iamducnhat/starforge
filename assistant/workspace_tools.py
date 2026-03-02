from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import queue
import json
from pathlib import Path
from typing import Any

from .utils import ensure_dir, read_json, redact_secrets_text, slugify, utc_now_iso, write_json, write_text
from .logging_config import get_logger

logger = get_logger(__name__)

class WorkspaceTools:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.plans_dir = self.workspace_root / "memory" / "plans"
        self._terminals: dict[str, dict[str, Any]] = {}
        ensure_dir(self.plans_dir)

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to workspace root with security validation."""
        raw = Path(path)
        try:
            resolved = raw.resolve() if raw.is_absolute() else (self.workspace_root / raw).resolve()
        except Exception as e:
            raise ValueError(f"invalid path: {path}") from e

        root_str = str(self.workspace_root)
        resolved_str = str(resolved)
        if resolved_str != root_str and not resolved_str.startswith(root_str + os.sep):
            raise ValueError(f"path outside workspace: {path}")
        return resolved

    def _to_workspace_rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except Exception:
            return str(path)

    def list_files(
        self,
        path: str = ".",
        glob: str = "**/*",
        include_hidden: bool = False,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        """List files in a directory with comprehensive validation and error handling."""
        
        # Validate input parameters
        if not isinstance(path, str):
            return {"ok": False, "error": "path must be a string"}
        if not isinstance(glob, str):
            return {"ok": False, "error": "glob must be a string"}
        if not isinstance(include_hidden, bool):
            return {"ok": False, "error": "include_hidden must be a boolean"}
        if not isinstance(max_entries, int) or max_entries < 1:
            return {"ok": False, "error": "max_entries must be a positive integer"}

        try:
            base = self._resolve(path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        if not base.exists():
            return {"ok": False, "error": f"path not found: {path}"}
        if not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}

        entries: list[dict[str, Any]] = []
        pattern = glob.strip() or "**/*"

        try:
            for p in base.glob(pattern):
                if len(entries) >= max(1, max_entries):
                    break

                rel = self._to_workspace_rel(p)
                if not include_hidden and any(part.startswith(".") for part in Path(rel).parts):
                    continue
                if p.is_dir():
                    continue

                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                entries.append({"path": rel, "size": size})

            entries.sort(key=lambda x: x["path"])
            return {"ok": True, "root": self._to_workspace_rel(base), "count": len(entries), "files": entries}
        except Exception as e:
            logger.error(f"Failed to list files in {path}: {e}")
            return {"ok": False, "error": f"error listing files: {str(e)}", "hint": "Check permissions and path validity"}

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = 12000,
    ) -> dict[str, Any]:
        file_path = self._resolve(path)
        if not file_path.exists():
            logger.warning(f"Read failed: file not found: {path}")
            return {"ok": False, "error": f"file not found: {path}"}
        if not file_path.is_file():
            logger.warning(f"Read failed: not a file: {path}")
            return {"ok": False, "error": f"not a file: {path}"}

        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        s = max(1, start_line)
        e = len(lines) if end_line is None else max(s, min(end_line, len(lines)))
        selected = "\n".join(lines[s - 1 : e])
        truncated = False
        if len(selected) > max_chars:
            selected = selected[: max_chars - 3] + "..."
            truncated = True

        masked = redact_secrets_text(selected)

        return {
            "ok": True,
            "path": self._to_workspace_rel(file_path),
            "start_line": s,
            "end_line": e,
            "truncated": truncated,
            "content": masked,
        }

    def create_file(self, path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        file_path = self._resolve(path)
        ensure_dir(file_path.parent)

        if file_path.exists() and not overwrite:
            logger.warning(f"Create failed: file already exists: {path}")
            return {
                "ok": False,
                "error": f"file already exists: {path}",
                "hint": "Use edit_file to update existing files, or create_file(..., overwrite=true) to replace.",
            }
        write_text(file_path, content)

        return {"ok": True, "path": self._to_workspace_rel(file_path), "bytes": len(content.encode("utf-8"))}

    def create_folder(self, path: str) -> dict[str, Any]:
        folder_path = self._resolve(path)
        if folder_path.exists():
            return {"ok": False, "error": f"path already exists: {path}"}
        
        try:
            ensure_dir(folder_path)
            return {"ok": True, "path": self._to_workspace_rel(folder_path)}
        except Exception as e:
            return {"ok": False, "error": f"failed to create folder: {e}"}

    def delete_file(self, path: str) -> dict[str, Any]:
        target_path = self._resolve(path)
        if not target_path.exists():
            return {"ok": False, "error": f"path not found: {path}"}
        
        try:
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
            return {"ok": True, "path": self._to_workspace_rel(target_path), "deleted": True}
        except Exception as e:
            return {"ok": False, "error": f"failed to delete: {e}"}

    def run_terminal(self, action: str, cmd: str | None = None, session_id: str = "default") -> dict[str, Any]:
        actions = {"start", "send", "read", "close"}
        if action not in actions:
            return {"ok": False, "error": f"invalid action '{action}', must be one of {actions}"}

        if action == "start":
            if session_id in self._terminals:
                return {"ok": False, "error": f"Session {session_id} already exists."}
            
            try:
                proc = subprocess.Popen(
                    ["/bin/bash"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(self.workspace_root)
                )
            except Exception as e:
                return {"ok": False, "error": f"failed to start session: {e}"}

            out_queue = queue.Queue()

            def reader(pipe, q):
                try:
                    for line in iter(pipe.readline, ""):
                        q.put(line)
                except Exception:
                    pass
                finally:
                    pipe.close()
            
            t = threading.Thread(target=reader, args=(proc.stdout, out_queue), daemon=True)
            t.start()
            self._terminals[session_id] = {"proc": proc, "queue": out_queue, "thread": t}
            return {"ok": True, "session_id": session_id, "message": f"Started session: {session_id}"}

        s = self._terminals.get(session_id)
        if not s:
            return {"ok": False, "error": f"Session {session_id} not found."}

        if action == "send":
            if not cmd:
                return {"ok": False, "error": "No command provided for 'send' action."}
            try:
                # Add newline if not present
                if not cmd.endswith("\n"):
                    cmd += "\n"
                s["proc"].stdin.write(cmd)
                s["proc"].stdin.flush()
                return {"ok": True, "session_id": session_id, "message": f"Sent command."}
            except Exception as e:
                return {"ok": False, "error": f"failed to send command: {e}"}

        if action == "read":
            lines = []
            time.sleep(0.1)  # Brief wait for fast outputs
            while not s["queue"].empty():
                lines.append(s["queue"].get_nowait())
            output = "".join(lines)
            return {"ok": True, "session_id": session_id, "output": output if output else "[No new output]"}

        if action == "close":
            try:
                s["proc"].terminate()
                del self._terminals[session_id]
                return {"ok": True, "session_id": session_id, "message": f"Closed session: {session_id}"}
            except Exception as e:
                return {"ok": False, "error": f"failed to close session: {e}"}

    def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        # Backward-compatible shim. Prefer create_file + edit_file in tool protocol.
        file_path = self._resolve(path)
        ensure_dir(file_path.parent)
        if append:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(content)
            return {
                "ok": True,
                "path": self._to_workspace_rel(file_path),
                "bytes": len(content.encode("utf-8")),
                "deprecated": True,
                "hint": "write_file is deprecated; prefer create_file/edit_file.",
            }
        result = self.create_file(path=path, content=content, overwrite=True)
        result["deprecated"] = True
        result["hint"] = "write_file is deprecated; prefer create_file/edit_file."
        return result

    def edit_file(
        self,
        path: str,
        find_text: str,
        replace_text: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        if not find_text:
            return {"ok": False, "error": "find_text must not be empty"}

        file_path = self._resolve(path)
        if not file_path.exists() or not file_path.is_file():
            return {
                "ok": False,
                "error": f"file not found: {path}",
                "hint": "Create the file first with create_file(path, content), then call edit_file.",
            }

        original = file_path.read_text(encoding="utf-8", errors="replace")
        if find_text not in original:
            return {"ok": False, "error": "find_text not found"}

        if replace_all:
            updated = original.replace(find_text, replace_text)
            replacements = original.count(find_text)
        else:
            updated = original.replace(find_text, replace_text, 1)
            replacements = 1

        write_text(file_path, updated)
        return {"ok": True, "path": self._to_workspace_rel(file_path), "replacements": replacements}

    def search_project(
        self,
        query: str,
        path: str = ".",
        glob: str = "**/*",
        case_sensitive: bool = False,
        regex: bool = False,
        max_matches: int = 200,
    ) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists():
            return {"ok": False, "error": f"path not found: {path}"}
        if not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}
        q = (query or "").strip()
        if not q:
            return {"ok": False, "error": "query must not be empty"}

        limit = max(1, min(int(max_matches), 1000))
        rel_base = self._to_workspace_rel(base)

        if shutil.which("rg"):
            cmd = ["rg", "--json", "--line-number", "--color", "never", "--max-count", str(limit)]
            if not case_sensitive:
                cmd.append("-i")
            if not regex:
                cmd.append("-F")
            if glob and glob != "**/*":
                cmd.extend(["-g", glob])
            cmd.extend([q, str(base)])
            try:
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                matches: list[dict[str, Any]] = []
                for line in proc.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if item.get("type") != "match":
                        continue
                    data = item.get("data", {})
                    path_text = str(data.get("path", {}).get("text", "")).strip()
                    line_num = int(data.get("line_number", 0) or 0)
                    text_data = data.get("lines", {}).get("text", "")
                    text_val = str(text_data).rstrip("\n")
                    if not path_text:
                        continue
                    try:
                        rel = self._to_workspace_rel(Path(path_text))
                    except Exception:
                        rel = path_text
                    matches.append({"path": rel, "line": line_num, "text": text_val})
                    if len(matches) >= limit:
                        break
                return {
                    "ok": True,
                    "engine": "rg",
                    "query": q,
                    "path": rel_base,
                    "count": len(matches),
                    "matches": matches,
                }
            except Exception:
                pass

        # Python fallback search
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(q, flags=flags) if regex else None
        matches: list[dict[str, Any]] = []
        for p in base.glob(glob.strip() or "**/*"):
            if len(matches) >= limit:
                break
            if not p.is_file():
                continue
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "\x00" in raw:
                continue
            for i, line in enumerate(raw.splitlines(), start=1):
                hit = bool(pattern.search(line)) if pattern else (q in line if case_sensitive else q.lower() in line.lower())
                if not hit:
                    continue
                matches.append({"path": self._to_workspace_rel(p), "line": i, "text": line[:400]})
                if len(matches) >= limit:
                    break

        return {
            "ok": True,
            "engine": "python",
            "query": q,
            "path": rel_base,
            "count": len(matches),
            "matches": matches,
        }

    def _plan_path(self, plan_id: str) -> Path:
        return self.plans_dir / f"{slugify(plan_id)}.json"

    def _load_plan(self, plan_id: str) -> dict[str, Any] | None:
        path = self._plan_path(plan_id)
        if not path.exists():
            return None
        try:
            return read_json(path)
        except Exception:
            return None

    def create_plan(self, title: str, goal: str, steps: list[str]) -> dict[str, Any]:
        clean_steps = [s.strip() for s in steps if isinstance(s, str) and s.strip()]
        if not clean_steps:
            return {"ok": False, "error": "steps must contain at least one item"}

        base_id = slugify(title) or "plan"
        plan_id = base_id
        idx = 2
        while self._plan_path(plan_id).exists():
            plan_id = f"{base_id}_{idx}"
            idx += 1

        now = utc_now_iso()
        todos = []
        for i, step in enumerate(clean_steps, start=1):
            todos.append(
                {
                    "id": i,
                    "text": step,
                    "status": "pending",
                    "created_at": now,
                    "updated_at": now,
                }
            )

        plan = {
            "id": plan_id,
            "title": title.strip() or plan_id,
            "goal": goal.strip(),
            "created_at": now,
            "updated_at": now,
            "todos": todos,
        }
        write_json(self._plan_path(plan_id), plan)
        return {"ok": True, "plan_id": plan_id, "todo_count": len(todos), "plan": plan}

    def list_plans(self) -> dict[str, Any]:
        plans: list[dict[str, Any]] = []
        for p in sorted(self.plans_dir.glob("*.json")):
            try:
                plan = read_json(p)
            except Exception:
                continue
            todos = plan.get("todos", [])
            pending = 0
            done = 0
            if isinstance(todos, list):
                for todo in todos:
                    if not isinstance(todo, dict):
                        continue
                    if todo.get("status") == "done":
                        done += 1
                    else:
                        pending += 1
            plans.append(
                {
                    "id": plan.get("id", p.stem),
                    "title": plan.get("title", p.stem),
                    "goal": plan.get("goal", ""),
                    "updated_at": plan.get("updated_at", ""),
                    "pending": pending,
                    "done": done,
                }
            )
        return {"ok": True, "count": len(plans), "plans": plans}

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        plan = self._load_plan(plan_id)
        if not plan:
            return {"ok": False, "error": f"plan not found: {plan_id}"}
        return {"ok": True, "plan": plan}

    def add_todo(self, plan_id: str, text: str) -> dict[str, Any]:
        plan = self._load_plan(plan_id)
        if not plan:
            return {"ok": False, "error": f"plan not found: {plan_id}"}

        todos = plan.get("todos", [])
        if not isinstance(todos, list):
            todos = []
        next_id = max([int(t.get("id", 0)) for t in todos if isinstance(t, dict)] + [0]) + 1
        now = utc_now_iso()
        todo = {"id": next_id, "text": text.strip(), "status": "pending", "created_at": now, "updated_at": now}
        todos.append(todo)
        plan["todos"] = todos
        plan["updated_at"] = now
        write_json(self._plan_path(plan_id), plan)
        return {"ok": True, "plan_id": plan_id, "todo": todo}

    def update_todo(self, plan_id: str, todo_id: int, status: str) -> dict[str, Any]:
        plan = self._load_plan(plan_id)
        if not plan:
            return {"ok": False, "error": f"plan not found: {plan_id}"}

        valid = {"pending", "in_progress", "done"}
        if status not in valid:
            return {"ok": False, "error": f"invalid status: {status}"}

        todos = plan.get("todos", [])
        if not isinstance(todos, list):
            return {"ok": False, "error": "invalid plan todos"}

        updated = None
        now = utc_now_iso()
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            if int(todo.get("id", -1)) == int(todo_id):
                todo["status"] = status
                todo["updated_at"] = now
                updated = todo
                break

        if updated is None:
            return {"ok": False, "error": f"todo not found: {todo_id}"}

        plan["updated_at"] = now
        write_json(self._plan_path(plan_id), plan)
        return {"ok": True, "plan_id": plan_id, "todo": updated}
