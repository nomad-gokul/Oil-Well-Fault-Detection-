# Live deployment (Task 20)

Free, two-part setup:

| Part | Where | Built from |
|---|---|---|
| **API** (FastAPI + model + demo replay) | [Render](https://render.com) free web service | `docker/api.Dockerfile` in this repo, via `render.yaml`; model in `deploy/bundle/` |
| **Dashboard** (Streamlit) | [Streamlit Community Cloud](https://share.streamlit.io) | `src/owfd/dashboard/app.py` in this repo |

The API runs in **demo mode**: it replays 10 held-out real wells from the 3W test set (one per
event class) at 30x speed, so visitors always see live predictions and alerts. Measured in
the production image: about 155 MB of memory and 5.5% of one CPU core on average (Render's free
plan: 512 MB, 0.1 CPU).

## What you do by hand (about 15 minutes)

### 0. Merge the pull request
Merge the `develop` → `main` pull request on GitHub (Render and Streamlit deploy from `main`).

### 1. API on Render
1. Sign in at <https://dashboard.render.com> with your GitHub account.
2. **New → Blueprint**, pick the repository `nomad-gokul/Oil-Well-Fault-Detection-`, branch `main`.
   Render reads `render.yaml` and proposes one web service, `owfd-api` (free plan). Click **Apply**.
3. Wait for the first build (5–10 minutes). The service URL looks like
   `https://owfd-api-xxxx.onrender.com`. Open `…/docs` to see the API.
4. In the service's **Environment** tab, copy the generated value of `OWFD_API_KEY`.
   (Optional) set `OWFD_CORS_ORIGINS` to your dashboard URL from step 2, e.g.
   `["https://oil-well-fault-detection.streamlit.app"]`.

### 2. Dashboard on Streamlit Community Cloud
1. Sign in at <https://share.streamlit.io> with GitHub → **Create app** → **Deploy a public app from GitHub**.
2. Repository `nomad-gokul/Oil-Well-Fault-Detection-`, branch `main`,
   main file path **`src/owfd/dashboard/app.py`**, choose an app URL
   (e.g. `oil-well-fault-detection`).
3. **Advanced settings**: Python **3.11**, and these **Secrets** (TOML):

   ```toml
   OWFD_API_URL = "https://owfd-api-xxxx.onrender.com"
   OWFD_API_KEY = "the value copied from Render"
   ```

   (The key enables the Acknowledge buttons and the demo controls; without it the dashboard
   is read-only.) Dependencies come from `src/owfd/dashboard/requirements.txt`.
4. **Deploy**. The first start takes a few minutes.

### 3. Check it
```bash
scripts/smoke_test.sh https://owfd-api-xxxx.onrender.com https://oil-well-fault-detection.streamlit.app
```
It checks `/health`, `/model`, `/docs`, a real `/predict`, that `/ingest` rejects requests
without the key, that the demo is running, `/metrics`, and the dashboard.

### 4. Keep it awake (optional, recommended)
Render's free services sleep after 15 minutes without traffic and need about a minute to wake
up (the dashboard shows "waking up" meanwhile). A free uptime monitor
(e.g. [UptimeRobot](https://uptimerobot.com)) pinging `https://owfd-api-xxxx.onrender.com/health`
every 5 minutes keeps it awake and alerts you if it goes down. One always-on free service fits
within Render's 750 free hours per month.

## Secrets and settings (names only)

| Variable | Where | Purpose |
|---|---|---|
| `OWFD_API_KEY` | Render (generated), Streamlit secret | protects `/ingest`, acknowledging alerts, demo start/stop |
| `OWFD_API_URL` | Streamlit secret | where the dashboard finds the API |
| `OWFD_DEMO_MODE`, `OWFD_DEMO_SPEED` | Render (`render.yaml`) | built-in replay of demo wells |
| `OWFD_RATE_LIMIT_PER_MIN` | Render | POST requests per client IP per minute (default 600) |
| `OWFD_CORS_ORIGINS` | Render | allowed browser origins (JSON list) |
| `OWFD_DATABASE_URL` | Render (optional) | Postgres for persistent alerts, e.g. a free Neon/Supabase database: `postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require` |
| `OWFD_ALERT_WEBHOOK_URL` | Render (optional) | Slack/Discord-compatible webhook for new alerts |
| `RENDER_DEPLOY_HOOK_URL` | GitHub repo secret (optional) | lets the release workflow trigger a Render deploy |

Without `OWFD_DATABASE_URL` the API uses SQLite inside the container: alerts and predictions
are lost when Render restarts or redeploys the service. That is fine for the demo.

## Lock-down

- Write endpoints (`/ingest`, `/alerts/{id}/ack`, `/demo/start|stop`) require `X-API-Key`.
- Rate limit on all POST endpoints per client IP (`429` with `Retry-After`).
- `/predict` is public (stateless scoring, rate-limited) so the API docs can be tried out.
- The image runs as a non-root user; no secrets are baked into images or the repository.

## Updating the model

```bash
python -m owfd.models.train_xgb ...      # or tune / evaluate (Tasks 7-10)
python -m owfd.models.export             # artifacts/serving/<version>/
make publish-bundle                      # copy to deploy/bundle/ (gzipped model, < 1 MB)
git add deploy/bundle && git commit -m "Update production model" && git push
```
CI rebuilds and smoke-tests the image with the committed model; Render redeploys `main`
automatically. Publishing a GitHub Release also pushes `owfd-api` and `owfd-dashboard` images
to GitHub Container Registry (`.github/workflows/release.yml`).

## Alternative: one small VM

On any VM with Docker (e.g. a 1 GB instance):
```bash
git clone https://github.com/nomad-gokul/Oil-Well-Fault-Detection-.git && cd Oil-Well-Fault-Detection-
OWFD_API_KEY=change-me docker compose up -d --build --wait
```
Put a reverse proxy with automatic HTTPS in front (e.g. Caddy:
`caddy reverse-proxy --from your.domain --to localhost:8501`).
