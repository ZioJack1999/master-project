# Project instructions for coding agents

## Start every session

1. Read `TODO.md`, `README.md` and `X24_CONNECTION_STATUS.md` before proposing work or giving launch instructions. These describe the current supported workflow. `RESUME_NOTES.md` is historical unless explicitly updated to say otherwise.
2. Inspect `git status`, current branch/remotes and recent commits. Fetch origin when available and compare changes before editing. Fast-forward only when safe; preserve local work and do not reset, force-push or restore retired files to resolve divergence.
3. Continue the highest-priority actionable item in `TODO.md` when the user asks to continue. Follow the user's current request when it changes priorities. Ask only for missing information that actually blocks the next step.
4. Distinguish facts verified locally, results documented by another session, user reports and unverified assumptions. A repository note about a lab test does not mean this workstation has the hardware or private recordings.

## Project objective and boundaries

- Validate person-specific motor-imagery neural decoding before Thanksgiving, November 26, 2026. Aim for at least 70% balanced accuracy on genuinely new validation cues; this is a target, not an established result.
- Keep binary (Rest / imagined right-hand grasp) separate from three-class (Rest / Volar Grasp / Finger Extension) calibration, models, sessions and reported results.
- Prioritize the three-class X24 workflow. Binary is the fallback. Keep the experimental 5-, 7- and 9-class ZIPs available pending a decision on lab use. Historical offline and cross-subject studies remain in Git history.
- Prevent leakage: training, tuning and validation must use separate whole cues/blocks. Never split overlapping windows from the same cue across those sets. Do not tune on validation and then call it unseen. Report rejected cues, class recall, confusion matrix, coverage and train/tuning gaps alongside balanced accuracy.
- Synthetic tests prove software behavior only. Historical participant-08 reference results are not new-wearer or live personal accuracy. Binary templates are untrained until personal calibration.

## Supported hardware and runtime

- Use `Launch_Studio.bat` and select the X24 bridge for the shared dashboard (port 8051). Three-class is the initial profile. The fixed-profile `Launch_THREECLASS_X24.bat` and `Launch_BINARY_X24.bat` (port 8052) remain available. These supervise bridge startup, fresh-stream checks and the GUI. Run one at a time, keep the launcher open, and save before stopping with Ctrl+C.
- For expanded validation, use `Launch_MULTICLASS_X24.bat` from the separately extracted ZIP (port 8055, 8057 or 8059). The expanded profiles start from scratch and save their own private sessions. Use the same single-bridge rule.
- Acquisition: existing 32-bit Python 3.9 `mastersp`, x86 Athena SDK at `C:\ABM\B-Alert\SDK`, through `athena_acquisition.py` and `x24_lsl_bridge.py`. Decoder: existing 64-bit conda Python through `run_x24.py`. See current docs for exact per-PC paths.
- Do not install 64-bit decoder requirements into `mastersp`, rewrite the backend, or create/modify environments unnecessarily. Verify the actual machine's dependencies first. Proprietary SDK, drivers and `.runtime/x24-lsl32` are not distributed in Git.
- Only one process may own the ESU. Do not launch hardware acquisition just to inspect code. Do not launch the old FES-initializing application. No new actuator/FES integration is in scope.
- Preserve SDK-owned `GetRawData` memory semantics: copy before the next SDK call; do not call `FreeBuffer` on that pointer. Keep units, named channel selection, sample-counter continuity and freshness guards explicit. Do not invent physical montage/reference mappings.
- Never substitute demo data for a requested live check. Hardware/display latency remains unmeasured despite verified sample transport.

## Changes and checks

- Preserve retained binary core/model fingerprints; run `verify_binary_release.py` for related work. If a requested binary change requires a new baseline, explain and version it explicitly rather than silently regenerating hashes.
- Use `verify_threeclass_bundle.py` for three-class runtime/reference closure and targeted tests for changed behavior. Run `python -m pytest -q` when warranted in an available development runtime. Do not start hardware-dependent checks without hardware and operator context.
- Keep raw EEG, participant identifiers, personal models, SDK files and logs private under ignored paths. Do not stage them or publish them in handoffs. Never commit credentials.
- User has requested GitHub push and Box/local delivery of completed fixes. Commit only intended reviewed changes; push normally when authorized and available. Sync deliberate deliverables without deleting unrelated recordings or overwriting divergent copies. State exactly what was checked and synced; local Box copies do not prove cloud upload completion.

## Finish every session

Update `TODO.md` before the final response, including a pause or incomplete session: current date, reviewed baseline, work completed, exact tests/results, remaining tasks in priority order, blockers, next concrete action, and running processes. Record commit/push/delivery status without claiming unperformed actions. Keep important prior decisions and evidence; replace stale current-state text rather than stacking contradictory instructions.

Update operator docs and `X24_CONNECTION_STATUS.md` when their facts change. Update this file if workflow rules change. Avoid recursive commit bookkeeping: the handoff may identify its own commit by subject and use Git history to resolve its hash.

These instructions are loaded by agents working in this repository. They do not schedule work or automatically open this project in an unrelated chat.
