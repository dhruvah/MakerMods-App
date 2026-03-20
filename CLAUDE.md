# MakerMods LeRobot UI

Web UI for xLeRobot SO101 bimanual robot arms — teleoperation, calibration, and data recording.

## Architecture

- **Backend**: FastAPI (Python) at `backend/` — wraps lerobot CLI commands via subprocess
- **Frontend**: Next.js 16 (App Router) at `frontend/` — wizard-style setup flow
- **State**: React Context (client) + `webui_config.json` (backend persistence)
- **Communication**: REST API + WebSocket for real-time log streaming
- Frontend proxies `/api/*` and `/ws/*` to the backend via Next.js rewrites in `frontend/next.config.ts`

## Prerequisites

- Python with lerobot installed (`conda activate lerobot`)
- Node.js for the frontend

## Running

```bash
# Terminal 1: Backend (port 8000)
conda activate lerobot
cd /path/to/MakerMods-LeRobot-UI
python -m backend.main

# Terminal 2: Frontend (port 3000)
cd frontend
npm install   # first time only
npm run dev

# Open: http://localhost:3000
```

## Backend

### Import conventions
- Internal imports use `from backend.*` (e.g. `from backend.services.config_manager import ConfigManager`)
- `lerobot.motors.*` imports reference the actual lerobot library (must be installed)

### Key paths
- `backend/main.py` — FastAPI app, CORS, router registration, static file mounts
- `backend/api/` — Route handlers (setup, calibration, teleoperation, recording, config, huggingface, system)
- `backend/models/` — Pydantic models (config, setup, system, recording, teleoperation)
- `backend/services/` — Business logic (config_manager, process_manager, port_scanner, camera_scanner, calibration_service, hf_service, manual_calibration, auto_calibration)
- `backend/websockets/logs.py` — WebSocket log streaming
- `webui_config.json` — Persisted config at repo root (gitignored)

### Path resolution
- `repo_root` in `main.py` = `Path(__file__).parent.parent` (repo root)
- `repo_root` in `config_manager.py` = `Path(__file__).parent.parent.parent` (repo root)

## Frontend

### Stack
- Next.js 16, React 19, TypeScript, Tailwind CSS v4, shadcn/ui (new-york style)
- Path aliases: `@/components`, `@/lib`, `@/hooks`

### Key paths
- `frontend/app/page.tsx` — Single-page wizard host
- `frontend/components/wizard/` — Wizard layout, provider (React Context + reducer), sidebar, topbar, step-card
- `frontend/components/wizard/steps/` — 6 steps: robot-type, ports, cameras, calibration, teleoperate, record
- `frontend/components/common/` — Shared components (log-viewer, process-status, dev-error-panel, robot-display)
- `frontend/components/ui/` — shadcn/ui components (do not edit manually, use `npx shadcn add`)
- `frontend/lib/api.ts` — Real API client
- `frontend/lib/services.ts` — Service layer (has `USE_MOCK` toggle)
- `frontend/lib/wizard-types.ts` — TypeScript interfaces, constants, initial state
- `frontend/hooks/` — Custom hooks (use-websocket, use-manual-calibration)

## Development notes

- **IMPORTANT**: After every feature request, bug fix, or code revamp, update `PROGRESS.md` with a dated changelog entry describing what changed and which files were modified. This is mandatory for every PR/commit.
- Backend wraps lerobot CLI commands via subprocess — zero changes to lerobot core code
- Don't cast Pydantic-derived interfaces with `as Record<string, unknown>` — use spread operator instead

## Bimanual mode — how IDs and calibration work

This is critical to understand. Lerobot's bimanual wrappers (`bi_so101_follower`, `bi_so101_leader`) use a **base ID** convention:

### ID derivation
- Commands use `--robot.id={base}` and `--teleop.id={base}` (e.g. `--robot.id=bimanual_follower`)
- Lerobot internally creates two sub-arm instances (`SO101Follower`/`SO101Leader`) with IDs `{base}_left` and `{base}_right`
- Example: `--robot.id=bimanual_follower` → sub-arms `bimanual_follower_left` and `bimanual_follower_right`
- Optional overrides: `--robot.left_id` / `--robot.right_id` bypass the derivation

### Calibration file storage
- Bimanual calibration files are stored under the **wrapper type directory**:
  - `~/.cache/huggingface/lerobot/calibration/robots/bi_so101_follower/{base}_left.json`
  - `~/.cache/huggingface/lerobot/calibration/robots/bi_so101_follower/{base}_right.json`
  - `~/.cache/huggingface/lerobot/calibration/teleoperators/bi_so101_leader/{base}_left.json`
  - `~/.cache/huggingface/lerobot/calibration/teleoperators/bi_so101_leader/{base}_right.json`
- Single-arm calibration files go under `so101_follower` / `so101_leader`

### How the UI handles this
- Backend `BimanualConfig` stores `follower_id` and `leader_id` (base IDs)
- Frontend calibration step lets users name each arm independently, but validates that left/right pairs share the same prefix with `_left`/`_right` suffixes
- `validateBimanualCalibrationNames()` in `wizard-types.ts` extracts the base ID from the naming pattern
- `saveConfig()` sends the derived base IDs to the backend
- Teleoperation/recording commands pass `--robot.id={follower_id}` and `--teleop.id={leader_id}`

### Lerobot CLI examples (for reference)
```bash
# Calibrate bimanual follower
lerobot-calibrate --robot.type=bi_so101_follower --robot.id=bimanual_follower \
  --robot.left_arm_port=/dev/tty.usbmodemXXXX --robot.right_arm_port=/dev/tty.usbmodemYYYY

# Teleoperate bimanual
lerobot-teleoperate --robot.type=bi_so101_follower --robot.id=bimanual_follower \
  --robot.left_arm_port=... --robot.right_arm_port=... \
  --teleop.type=bi_so101_leader --teleop.id=bimanual_leader \
  --teleop.left_arm_port=... --teleop.right_arm_port=...

# Record bimanual dataset
lerobot-record --robot.type=bi_so101_follower --robot.id=bimanual_follower \
  --robot.left_arm_port=... --robot.right_arm_port=... \
  --teleop.type=bi_so101_leader --teleop.id=bimanual_leader \
  --teleop.left_arm_port=... --teleop.right_arm_port=... \
  --dataset.repo_id=user/dataset --dataset.single_task="task" --dataset.num_episodes=50
```
