# DSAA Frontend

React + TypeScript + Vite shell for the existing FastAPI service in `llm_engine/`.
This foundation currently reads backend health and model metadata; prompt generation
and streaming are intentionally not part of this phase.

## Run locally

Start the backend from the repository root:

```powershell
python -m llm_engine.example_server
```

In another terminal, from `frontend/`:

```powershell
npm install
npm run dev
```

Open the Vite URL printed in the terminal (usually `http://localhost:5173`). The
development proxy forwards `/api/*` to `http://127.0.0.1:8000/*`.

`VITE_API_BASE_URL` controls the API base path. The default is `/api`, which uses
the Vite proxy in development. Copy `.env.example` to `.env.local` to override it.
For a deployed frontend, configure this value to the backend origin or the path
served by the deployment reverse proxy.

## Checks and production build

```powershell
npm run typecheck
npm run build
npm run preview
```

The backend health and models endpoints must be reachable for runtime status to
show as connected. The status panel refreshes every 15 seconds and distinguishes
API reachability from whether model weights have loaded.
