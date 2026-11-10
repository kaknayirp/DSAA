# DSAA Studio

Start the [FastAPI server](../backend/README.md) first. In a second PowerShell terminal, run:

```powershell
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite. Overview shows live host telemetry, Playground runs a registered strategy, and Cache Explorer displays the latest or a loaded run. Compare Modes requires Original and DSAA. Benchmark Lab queues recall evaluations once the evaluation team's adapter is connected, and Reports downloads completed experiments. Algorithm Tuning saves settings for future DSAA runs.

The explorer and comparison views include a labeled sample. The overview and benchmark views show measured data only. Unconnected strategies are marked unavailable.

`npm run build` type-checks and builds the application. A host serving the built assets should forward same-origin `/api` and `/ws` traffic to FastAPI.
