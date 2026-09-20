# GPU service split

- Remote: existing private `Good-Badminton-Local` repository.
- Worktree branch: `chore/gpu-service-split`, based on
  `932a0d2a54d4bc8e5a9d7341118c8d6ea8f0e854`.
- Retained: `api/`, `apps/`, `badminton_analysis/`,
  `good_badminton_contracts/`, GPU deployment scripts, and GPU-only tests.
- Removed: venue gateway code, business API/database code, Gradio UI, Next.js
  operator UI, and evaluation-only scripts/tests.
- Adaptation: complete-video anonymous candidate crops moved from the old UI
  helper into `api/candidate_photos.py`; no identity data is added.
