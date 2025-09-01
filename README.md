
# Production Scheduler (CP-SAT) — Streamlit App

A runnable demonstration of a discrete-time production scheduler with phased operations (setup/run/clean), machine capacity, worker skills/calendars, and precedence. Built with **OR-Tools** (Google) CP-SAT and Streamlit.

## Acronyms & Terms
- **CP-SAT**: Constraint Programming with a SAT-based solver.
- **OR-Tools**: Open-source optimization toolkit by Google.
- **DAG**: Directed Acyclic Graph (for operation precedence).
- **KPI**: Key Performance Indicator (makespan, tardiness, etc.).
- **Gantt**: Timeline chart showing resource allocation over time.

## Install & run
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## How to use
1. Keep the default JSON dataset (or edit it).
2. Click **Solve**.
3. Inspect KPIs, chosen machines, and schedules (tables and Gantt-like chart).
4. Tweak: durations, eligibility, due dates, worker skills, calendars, and re-run.

## Notes
- Fractional worker attention is supported by scaling with `CAP` (default 10 → 0.1 increments).
- This basic model uses bucketed time and exclusive machine choice per operation (no parallel splitting).
- Extend by adding sequence-dependent setup costs, sublots for parallel run, and handover penalties.
