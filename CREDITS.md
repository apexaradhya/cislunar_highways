# Credits

This project was built with **Claude Code**, Anthropic's AI coding assistant (https://claude.ai). Claude wrote nearly all of the code and documentation. This file says exactly who did what.

## Who did what

| Part | Who |
|---|---|
| Project idea, goals and specification (`PROJECT_SPEC.md`) | Me |
| Choosing the scope and the additions (design sweep, predictability study, cinematic explorer) | Me |
| Reviewing results, presenting and defending the project | Me |
| Physics model, integrators, root finder, STM (`src/cislunar/`) | Claude |
| Lagrange points, Hill regions, periodic orbits, the NRHO | Claude |
| Invariant manifolds, Poincaré sections, the heteroclinic connection | Claude |
| Transfer design, the design-space sweep, the predictability study | Claude |
| Test suite (106 tests) and the validation against SciPy | Claude |
| All figures, the animations, the three.js explorer and the video | Claude |
| Documentation: README, math notes, results, Q&A prep, learning notes, `PROGRESS.md` | Claude |
| Packaging, Makefile, GitHub Actions workflow | Claude |

## How it was made

I gave Claude a detailed specification. It then worked through the project in phases. Each phase got:
- the code and its tests;
- figures;
- a short learning note;
- an entry in `PROGRESS.md`.

The core numerical methods were written from scratch rather than taken from libraries. SciPy is used only to cross-check them and for one optimizer. `PROGRESS.md` records the problems that came up along the way and how each was fixed.

## Other credits

- **Textbooks and papers** are cited in [`docs/references.md`](docs/references.md).
- **three.js** (MIT license) draws the 3D explorer.
- **NumPy, SciPy, Matplotlib and Plotly** do the numerical work and the plotting.
