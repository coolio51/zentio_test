
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


## Test data

{
    "Machines": ["M1", "M2"],
    "Workers": ["W1", "W2", "W3", "W4", "W5"],
    "Skills": ["Setup", "Run", "Clean"],
    "Phases": ["Setup", "Run", "Clean"],
    "T": [1,2,3,4,5,6,7,8,9,10],
    "Ops": ["OpA", "OpB", "OpC"],
    "JobOf": {"OpA": "J1", "OpB": "J1", "OpC": "J2"},
    "Jobs": ["J1", "J2"],
    "Due": {"J1": 10, "J2": 6},
    "E": {
      "OpA": {"M1": 1, "M2": 0},
      "OpB": {"M1": 1, "M2": 1},
      "OpC": {"M1": 0, "M2": 1}
    },
    "D": {
      "OpA": {"Setup": {"M1": 1}, "Run": {"M1": 3}, "Clean": {"M1": 1}},
      "OpB": {"Setup": {"M1": 1, "M2": 1}, "Run": {"M1": 3, "M2": 3}, "Clean": {"M1": 1, "M2": 1}},
      "OpC": {"Setup": {"M2": 1}, "Run": {"M2": 2}, "Clean": {"M2": 1}}
    },
    "Need": {
      "OpA": {"Setup": {"Setup": 1.0}, "Run": {"Run": 1.0}, "Clean": {"Clean": 1.0}},
      "OpB": {"Setup": {"Setup": 1.0}, "Run": {"Run": 1.0}, "Clean": {"Clean": 1.0}},
      "OpC": {"Setup": {"Setup": 1.0}, "Run": {"Run": 1.0}, "Clean": {"Clean": 1.0}}
    },
    "Q": {
      "W1": {"Setup": 1, "Run": 1, "Clean": 1},
      "W2": {"Setup": 0, "Run": 1, "Clean": 0},
      "W3": {"Setup": 0, "Run": 0, "Clean": 1},
      "W4": {"Setup": 1, "Run": 1, "Clean": 1},
      "W5": {"Setup": 0, "Run": 1, "Clean": 0}
    },
    "A_M": {
      "M1": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1, "10": 1},
      "M2": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1, "10": 1}
    },
    "A_W_frac": {
      "W1": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0, "7": 1.0, "8": 1.0, "9": 1.0, "10": 1.0},
      "W2": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0, "7": 1.0, "8": 1.0, "9": 1.0, "10": 1.0},
      "W3": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0, "7": 1.0, "8": 1.0, "9": 1.0, "10": 1.0},
      "W4": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0, "7": 1.0, "8": 1.0, "9": 1.0, "10": 1.0},
      "W5": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0, "7": 1.0, "8": 1.0, "9": 1.0, "10": 1.0}
    },
    "PredEdges": [["OpA", "OpB"]],
    "CAP": 10 }