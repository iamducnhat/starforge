from starforge import run


result = run(
    objective="debug the failing demo project tests and capture the failing output",
    context={
        "working_dir": "./demo_project",
        "constraints": ["stay inside the demo project"],
        "diagnostic_command": "pytest -q",
        "output_path": "debug_report.md",
    },
    config={
        "adapter": "code",
        "max_steps": 4,
        "mode": "autonomous",
    },
)

print(result)
