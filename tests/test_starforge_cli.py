from __future__ import annotations

import json
from unittest.mock import patch

from starforge.cli import main


def _fake_result() -> dict:
    return {
        "success": True,
        "steps": 2,
        "confidence": 0.77,
        "actions": [
            {
                "tool": "list_files",
                "status": "completed",
                "observation": {"type": "filesystem_snapshot"},
            },
            {
                "tool": "read_file",
                "status": "completed",
                "observation": {"type": "file_read"},
            },
        ],
        "result": {
            "human_readable": "Paragraph one.\n\nParagraph two.",
        },
    }


def test_run_default_prints_json(capsys) -> None:
    with patch("starforge.cli.run", return_value=_fake_result()):
        code = main(["run", "demo objective"])

    captured = capsys.readouterr()
    assert code == 0
    parsed = json.loads(captured.out)
    assert parsed["success"] is True
    assert parsed["steps"] == 2


def test_run_human_readable_prints_human_readable_summary(capsys) -> None:
    with patch("starforge.cli.run", return_value=_fake_result()):
        code = main(["run", "demo objective", "--human-readable", "--no-ansi"])

    captured = capsys.readouterr()
    assert code == 0
    assert "Paragraph one." in captured.out
    assert "Actions:" in captured.out
    assert "1. list_files [completed] -> filesystem_snapshot" in captured.out
    assert '"success": true' not in captured.out.lower()


def test_run_forwards_model_feedback_options() -> None:
    with patch("starforge.cli.run", return_value=_fake_result()) as mocked:
        code = main(
            [
                "run",
                "demo objective",
                "--model-feedback",
                "--model-name",
                "qwen2.5:7b",
                "--model-provider",
                "ollama",
                "--model-orchestrated",
            ]
        )

    assert code == 0
    _, kwargs = mocked.call_args
    config = kwargs["config"]
    assert config["model_feedback"] is True
    assert config["model_name"] == "qwen2.5:7b"
    assert config["feedback_provider"] == "ollama"
    assert config["model_orchestrated"] is True


def test_run_human_stream_prints_live_steps(capsys) -> None:
    def _side_effect(*args, **kwargs):  # noqa: ANN002, ANN003 - test helper
        del args
        on_action = kwargs["config"].get("on_action")
        assert callable(on_action)
        on_action(
            {
                "index": 1,
                "tool": "list_files",
                "status": "completed",
                "observation_type": "filesystem_snapshot",
            }
        )
        return _fake_result()

    with patch("starforge.cli.run", side_effect=_side_effect):
        code = main(["run", "demo objective", "--human-readable", "--stream", "--no-ansi"])

    captured = capsys.readouterr()
    assert code == 0
    assert "[step 1] list_files [completed] -> filesystem_snapshot" in captured.out


def test_run_json_stream_prints_live_steps_to_stderr(capsys) -> None:
    def _side_effect(*args, **kwargs):  # noqa: ANN002, ANN003 - test helper
        del args
        on_action = kwargs["config"].get("on_action")
        assert callable(on_action)
        on_action(
            {
                "index": 1,
                "tool": "read_file",
                "status": "completed",
                "observation_type": "file_read",
            }
        )
        return _fake_result()

    with patch("starforge.cli.run", side_effect=_side_effect):
        code = main(["run", "demo objective", "--stream"])

    captured = capsys.readouterr()
    assert code == 0
    assert "[step 1] read_file [completed] -> file_read" in captured.err
    parsed = json.loads(captured.out)
    assert parsed["success"] is True


def test_run_no_stream_disables_live_updates() -> None:
    def _side_effect(*args, **kwargs):  # noqa: ANN002, ANN003 - test helper
        del args
        assert kwargs["config"].get("on_action") is None
        return _fake_result()

    with patch("starforge.cli.run", side_effect=_side_effect):
        code = main(["run", "demo objective", "--human-readable", "--no-stream", "--no-ansi"])

    assert code == 0


def test_run_human_ansi_prints_ansi_sequences(capsys) -> None:
    with patch("starforge.cli.run", return_value=_fake_result()):
        code = main(["run", "demo objective", "--human-readable", "--ansi"])

    captured = capsys.readouterr()
    assert code == 0
    assert "\x1b[" in captured.out
