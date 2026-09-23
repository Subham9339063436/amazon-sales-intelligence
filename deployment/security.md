# Security Notes

Threat model for this portfolio project: a public demo on a small cloud
instance serving an unauthenticated REST API and dashboard.

## 1. Secrets policy

- **No secrets exist in this repository.** No AWS keys, tokens, passwords, or
  `.env` files are committed, and the Docker image contains none.
- `.gitignore` excludes `.env*`, `*.pem`, `*.key`, virtualenvs and caches.
- `.dockerignore` mirrors that: `.env*`, `*.pem`, `*.key`, `*-secret*`,
  `venv/`, `.git` — so credentials can't slip into the build context.
- The Dockerfile uses no `ENV`/`ARG` secrets; the app needs **no
  credentials** to run — data and model artifacts ship with the repository.
- If credentials are ever added: inject at runtime only
  (`docker run --env-file .env`, EC2 user-data, SSM parameters) — never
  commit, never bake into images.

## 2. Network exposure

- **Both services are unauthenticated by design** (portfolio scope):
  - `GET /health`, `GET /model`, `POST /predict` — no auth token,
  - Streamlit dashboard — no auth.
- Therefore: open security-group ports **22, 5001, 8501 to your IP only**,
  never `0.0.0.0/0` (except SSH from a known IP, ideally key-only).
- Safer pattern for anything long-lived:
  `docker run -p 127.0.0.1:5001:5001 ...` + reverse proxy (nginx/Caddy)
  with TLS and basic auth or OAuth in front.
- The API only accepts `application/json` bodies with strict schema
  validation (400/415 otherwise) — it never `eval`s/deserializes
  pickles from the request; the checkpoint is a `torch.save` file loaded
  server-side only.

## 3. Input handling (API)

- Request size: JSON bodies are validated feature-by-feature — missing,
  unknown, non-numeric, boolean, and non-finite values are rejected with
  HTTP 400 before any model call.
- Errors return JSON, never stack traces (`debug=False`; gunicorn default).
- The model path and metrics path come from `src/config.py`, not from
  request input — no path traversal surface.

## 4. Dependencies & images

- Base image `python:3.13-slim` (official, regularly updated); rebuild
  periodically to pick up patches.
- Python deps pinned by `requirements.txt` (lower bounds); rebuild +
  rerun `pytest tests/ -q` after bumping.
- Runs as root inside the container (default) — acceptable for a local
  demo; for production add a non-root `USER` and read-only mounts.

## 5. Host hardening checklist (EC2)

- [ ] SSH: key-only, source IP restricted, no password auth
- [ ] Security group: 22/5001/8501 limited to known IPs
- [ ] `apt-get upgrade` applied; reboot if kernel updated
- [ ] Docker daemon current; no published ports on `0.0.0.0` unintentionally
- [ ] Terminate idle instances; snapshot only what's needed
- [ ] No secrets in shell history, repo, image, or logs

## 6. Data note

The dataset is public sales-style data with **synthetic** customer contact
fields (generated with Faker during normalization) — no PII concerns for
display; still avoid presenting it as real customer data.
