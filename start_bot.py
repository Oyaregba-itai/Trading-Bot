"""
Safe bot launcher — kills any existing bot process before starting.
Run this instead of bot.py directly.
"""
import os, sys, subprocess, time, psutil

PID_FILE = os.path.join(os.path.dirname(__file__), ".bot.pid")
BOT_SCRIPT = os.path.join(os.path.dirname(__file__), "bot.py")

def kill_old():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                old_pid = int(f.read().strip())
            if psutil.pid_exists(old_pid):
                p = psutil.Process(old_pid)
                p.terminate()
                p.wait(timeout=5)
                print(f"Stopped old bot process (PID {old_pid})")
        except Exception:
            pass
        os.remove(PID_FILE)

def start():
    kill_old()
    print("Starting bot...")
    proc = subprocess.Popen(
        [sys.executable, BOT_SCRIPT],
        cwd=os.path.dirname(__file__),
        stdout=open(os.path.join(os.path.dirname(__file__), "bot_output.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    print(f"Bot started (PID {proc.pid})")
    print("Logs: bot_output.log")
    return proc

if __name__ == "__main__":
    proc = start()
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("Bot stopped.")
