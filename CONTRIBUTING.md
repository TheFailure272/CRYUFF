# Contributing to C.R.U.Y.F.F.

Thank you for your interest in contributing. This document explains how to set up your environment, submit changes, and follow the project's conventions.

---

## Quick Start

```bash
# Clone
git clone https://github.com/TheFailure272/CRYUFF.git
cd CRYUFF

# Backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

# Frontend
cd frontend
npm install
cd ..

# Pre-commit hooks
pip install pre-commit
pre-commit install

# Verify
python -m pytest tests/ -v    # 76 tests, < 3s
cd frontend && npm run build  # < 4s, 0 errors
```

---

## Development Workflow

1. **Branch** from `main`: `git checkout -b feature/your-feature`
2. **Write code** following existing patterns
3. **Add tests** in `tests/` — every engine must have test coverage
4. **Run checks**:
   ```bash
   python -m pytest tests/ -v
   cd frontend && npm run build
   ```
5. **Commit** with conventional commit messages:
   - `feat:` new feature
   - `fix:` bug fix (reference fix number, e.g. `fix F70:`)
   - `docs:` documentation only
   - `test:` adding tests
   - `refactor:` code restructuring
6. **Push** and open a Pull Request

---

## Code Conventions

### Python (Backend)
- **Python 3.12+** with `from __future__ import annotations`
- **Type hints** on all public functions
- **Docstrings** (NumPy style) on all classes and public methods
- **dataclass(slots=True)** for engine state
- **async/await** for all I/O operations
- Constants in `UPPER_SNAKE_CASE`, private helpers prefixed with `_`
- JAX code must use `jnp.where` / `lax.fori_loop` (no Python `if`/`for` on traced values)

### JavaScript (Frontend)
- **React** functional components with hooks
- **R3F** (React Three Fiber) for all 3D rendering
- Hooks in `src/hooks/`, utilities in `src/lib/`, components in `src/components/`

### Fixes
- Every fix is numbered sequentially: **F1, F2, ... F69**
- Each fix requires: a comment in the code, a test, and a `CHANGELOG.md` entry
- Reference the fix number in the commit message

---

## Architecture Rules

| Rule | Rationale |
|---|---|
| No `float()` cast on JAX-traced values | Breaks JIT compilation |
| No Python `if` on array values in JAX code | Use `jnp.where` or `lax.cond` |
| All WebSocket routes in `server/main.py` | Single gateway pattern |
| All engine state in `engine/` | Stateless server, stateful engines |
| GPS coordinates → SpatialBridge → pitch meters | Never use raw WGS84 in analysis |
| All dressing room URLs on 192.168.x.x | Air-gapped LAN (Fix F38) |
| Match time resolved via Redis `cruyff:match_clock` | Never use wall-clock for clips (Fix F44) |

---

## Testing

```bash
# Full suite
python -m pytest tests/ -v

# Single module
python -m pytest tests/test_setpiece_solver.py -v

# Frontend build check
cd frontend && npm run build
```

All PRs must pass the CI pipeline (`.github/workflows/ci.yml`).

---

## Need Help?

Open an issue describing your question or proposed change. For major architectural changes, open a discussion first.
