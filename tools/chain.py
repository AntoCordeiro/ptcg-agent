"""Run commands sequentially (GPU jobs one at a time). Usage: python tools/chain.py "cmd1" "cmd2" ..."""
import subprocess, sys, shlex, time
for c in sys.argv[1:]:
    args = shlex.split(c)
    args = [sys.executable if a == "python" else a for a in args]
    t = time.time(); print(f"=== RUN {c}", flush=True)
    rc = subprocess.call(args)
    print(f"=== END rc={rc} {time.time()-t:.0f}s {c}", flush=True)
