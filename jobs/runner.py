"""Tiny file-based job runner. Start once on Windows inside the conda env:
    python jobs\\runner.py
It watches jobs/queue/*.json, runs each job as a subprocess (in parallel),
streams output to jobs/logs/<id>.log and writes jobs/done/<id>.json on exit.
Job file format: {"id": "name", "cmd": ["python", "script.py", "--arg"], "cwd": "."}
"python" is replaced by this interpreter so jobs use the same conda env.
To stop a job, create an empty file jobs/kill/<id>.  Ctrl+C stops the runner.
"""
import json, os, subprocess, sys, time, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(ROOT, "jobs")
running = {}

def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(J, "runner.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")

def heartbeat():
    with open(os.path.join(J, "heartbeat.json"), "w") as f:
        json.dump({"time": time.time(), "running": list(running)}, f)

log(f"runner started, python={sys.executable}")
try:
    import torch
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''} cpus={os.cpu_count()}")
except Exception as e:
    log(f"torch import failed: {e}")

while True:
    for fn in sorted(os.listdir(os.path.join(J, "queue"))):
        if not fn.endswith(".json"):
            continue
        src = os.path.join(J, "queue", fn)
        try:
            job = json.load(open(src, encoding="utf-8"))
        except Exception:
            continue  # partially written, retry next loop
        jid = job.get("id", fn[:-5])
        os.replace(src, os.path.join(J, "running", fn))
        cmd = [sys.executable if c == "python" else c for c in job["cmd"]]
        cwd = os.path.join(ROOT, job.get("cwd", "."))
        out = open(os.path.join(J, "logs", jid + ".log"), "w", encoding="utf-8", buffering=1)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **job.get("env", {}))
        p = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, env=env)
        running[jid] = (p, out, fn, time.time())
        log(f"START {jid}: {' '.join(job['cmd'])}")
    for jid in list(running):
        p, out, fn, t0 = running[jid]
        if os.path.exists(os.path.join(J, "kill", jid)):
            p.kill(); os.remove(os.path.join(J, "kill", jid)); log(f"KILL {jid}")
        rc = p.poll()
        if rc is not None:
            out.close()
            os.replace(os.path.join(J, "running", fn), os.path.join(J, "done", fn))
            with open(os.path.join(J, "done", jid + ".status.json"), "w") as f:
                json.dump({"id": jid, "returncode": rc, "seconds": round(time.time() - t0, 1)}, f)
            log(f"END {jid} rc={rc} ({time.time()-t0:.0f}s)")
            del running[jid]
    heartbeat()
    time.sleep(2)
