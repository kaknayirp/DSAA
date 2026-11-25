# DSAA Frontend

React + TypeScript + Vite frontend for the existing FastAPI service in
`llm_engine/`. The chat workspace sends prompts to the existing `POST /generate`
endpoint and reads its Server-Sent Events response. No inference or response
simulation is implemented in the frontend.

The backend accepts one prompt string per generation request. Earlier turns stay
visible in the transcript, but the current API does not accept conversation
history, so they are not included in later requests. Generation settings are
sent using the fields supported by the backend schema. The active model appears
in the backend status panel; the model is loaded lazily on the first generation
request.

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
API reachability from whether model weights have loaded. Generation uses POST
SSE through the Vite `/api` proxy; cancellation calls the existing request cancel
endpoint after the stream supplies a request ID, and also closes the browser
stream with `AbortController`.
