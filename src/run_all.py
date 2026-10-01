"""Rebuild every result, table and figure from the saved data, in order.

Downloads are separate (fetch_prices, fetch_rates, fetch_benchmarks,
fetch_phase2) because Yahoo revises adjusted prices over time; everything
downstream of data/ is deterministic. Each step writes its report to
output/<step>_report.txt. Run from src/:  python run_all.py
"""

import subprocess
import sys
import time

from config import OUTPUT_DIR

STEPS = [
    ("data_checks", "check_data.py"),          # Stage 1
    ("stage2", "run_stage2.py"),               # signals, volatility, predictive regressions
    ("stage3", "run_stage3.py"),               # backtest engine, V1
    ("stage4", "run_stage4.py"),               # volatility targeting, benchmarks
    ("stage5", "run_stage5.py"),               # regime overlay (about 5 minutes)
    ("stage6", "run_stage6.py"),               # evaluation, crisis, costs, robustness, DSR
    ("data_checks_phase2", "check_data_phase2.py"),
    ("stage12", "run_stage12.py"),             # carry signals
    ("stage13", "run_stage13.py"),             # phase 2 portfolios
    ("tables", "make_tables.py"),
    ("figures", "make_figures.py"),
]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, script in STEPS:
        start = time.time()
        with open(OUTPUT_DIR / f"{name}_report.txt" if name not in ("data_checks", "data_checks_phase2")
                  else OUTPUT_DIR / f"{name}.txt", "w", encoding="utf-8") as log:
            done = subprocess.run([sys.executable, "-u", script], stdout=log, stderr=subprocess.STDOUT)
        print(f"{script:24s} exit {done.returncode}  {time.time() - start:6.1f}s")
        if done.returncode != 0:
            sys.exit(f"{script} failed; see output/{name} report")


if __name__ == "__main__":
    main()
