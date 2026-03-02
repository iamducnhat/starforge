import subprocess
def execute_terminal_command(command, stdin_input=None, timeout=60):
    try:
        result = subprocess.run(command, input=stdin_input, capture_output=True, text=True, shell=True, timeout=timeout)
        return {'stdout': result.stdout, 'stderr': result.stderr, 'returncode': result.returncode}
    except Exception as e:
        return {'error': str(e)}
