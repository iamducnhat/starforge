import subprocess, threading, time, queue

def terminal_session(action: str, cmd: str = None, session_id: str = 'default'):
    if '_TERM_SESS' not in globals():
        globals()['_TERM_SESS'] = {}
    sessions = globals()['_TERM_SESS']

    if action == 'start':
        if session_id in sessions: return f'Session {session_id} already exists.'
        proc = subprocess.Popen(
            ['/bin/bash'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        out_queue = queue.Queue()
        def reader(pipe, q):
            try:
                for line in iter(pipe.readline, ''):
                    q.put(line)
            except Exception: pass
            finally: pipe.close()
        
        t = threading.Thread(target=reader, args=(proc.stdout, out_queue), daemon=True)
        t.start()
        sessions[session_id] = {'proc': proc, 'queue': out_queue, 'thread': t}
        return f'Started session: {session_id}'

    s = sessions.get(session_id)
    if not s: return f'Session {session_id} not found.'

    if action == 'send':
        if not cmd: return 'No command.'
        s['proc'].stdin.write(cmd + '\n')
        s['proc'].stdin.flush()
        return f'Sent: {cmd}'

    if action == 'read':
        lines = []
        time.sleep(0.1) # Brief wait for output
        while not s['queue'].empty():
            lines.append(s['queue'].get_nowait())
        return ''.join(lines) if lines else '[No new output]'

    if action == 'close':
        s['proc'].terminate()
        del sessions[session_id]
        return f'Closed session: {session_id}'
