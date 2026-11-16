"""Chay main.py va ghi lai tien trinh nao gui SIGTERM (PID, lenh, chuoi cha) truoc khi thoat.

Dung: python exps/lbr/run_guard.py <config.json> <file_log_tin_hieu>
SIGTERM bi chan o moi luong (dat truoc khi tao luong / tien trinh con) va chi duoc nhan boi luong cho sigwaitinfo.
"""
import datetime
import os
import runpy
import signal
import sys
import threading

cfg, log = sys.argv[1], sys.argv[2]
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGHUP})


def cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:300]
    except OSError:
        return "?"


def ppid(pid):
    try:
        return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def waiter():
    info = signal.sigwaitinfo({signal.SIGTERM, signal.SIGINT, signal.SIGHUP})
    chain, p = [], info.si_pid
    for _ in range(6):
        if p <= 1:
            break
        chain.append(f"{p}: {cmdline(p)}")
        p = ppid(p)
    with open(log, "a") as f:
        f.write(f"{datetime.datetime.now()} nhan tin hieu {info.si_signo} tu pid {info.si_pid} uid {info.si_uid}\n")
        for line in chain:
            f.write(f"    {line}\n")
    os._exit(128 + info.si_signo)


threading.Thread(target=waiter, daemon=True).start()
sys.path.insert(0, os.getcwd())          # thu muc goc repo (nhu khi chay python main.py)
sys.argv = ["main.py", "--config", cfg]
runpy.run_path("main.py", run_name="__main__")
