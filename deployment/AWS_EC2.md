# AWS EC2 Deployment Guide

> **Status: DEPLOYMENT-READY — NOT DEPLOYED.**
> This document describes how to deploy the project. No deployment has been
> performed from this repository, and no AWS credentials exist in the repo.
> Follow these steps yourself in the AWS console/CLI; nothing here runs
> automatically.

The project ships as a single Docker image that serves both processes:

| Process | Default command | Port |
|---|---|---|
| Flask REST API | `gunicorn --bind 0.0.0.0:5001 api.app:app` | 5001 |
| Streamlit dashboard | `streamlit run app.py` (command override) | 8501 |

## 1. EC2 setup

1. AWS Console → **EC2 → Launch instance**:
   - **AMI:** Ubuntu Server 24.04 LTS (or Amazon Linux 2023)
   - **Instance type:** `t3.small` recommended (2 vCPU / 2 GiB).
     `t3.micro` (1 GiB) is free-tier eligible but tight — the API loads a
     PyTorch model (~300–500 MB RSS) and the dashboard loads pandas; run
     **one process at a time** on a micro.
   - **Key pair:** create/download a `.pem`, keep it out of git
     (`*.pem` is already in `.gitignore`/`.dockerignore`).
   - **Storage:** 20 GB gp3 (the image is ~2.3 GB plus Docker overhead).

2. **Security group** — see [security.md](security.md); minimum inbound
   rules: SSH (22) from your IP only, plus 5001/8501 if you want direct
   access (put a reverse proxy with TLS in front for anything public).

3. Connect:
   ```bash
   chmod 400 your-key.pem
   ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
   ```

## 2. Required packages (host)

Only Docker is needed on the host — Python dependencies live inside the image.

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y git ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER   # then log out/in
```

## 3. Clone the repository

```bash
git clone <YOUR_REPO_URL>.git
cd <repo-directory>
```

The trained checkpoint (`models/`) and evaluation artifacts (`results/`)
are part of this repository, so **no training run is required** to
deploy. To regenerate them instead:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/run_training.py        # Phase 1 metrics
python scripts/run_torch_training.py  # Phase 3 checkpoint + metrics
```

## 4. Docker setup

```bash
docker build -t amazon-sales-forecasting .
```

## 5. Environment variables

The app reads a single optional variable — no secrets are required:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5001` | API port when running `python api/app.py` (gunicorn uses `--bind`) |

Never place AWS keys or tokens in the image or in `docker run` flags for
this project — it needs none. If you add any later, inject them at runtime
(`--env-file .env`), never with `ENV`/`ARG` in the Dockerfile.

## 6. Starting the API

```bash
docker run -d --name sales-api -p 5001:5001 amazon-sales-forecasting
```

## 7. Starting the dashboard

```bash
docker run -d --name sales-dashboard -p 8501:8501 amazon-sales-forecasting \
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Or both at once with Compose: `docker compose up -d`.

## 8. Firewall / security-group considerations

- Inbound **22**: your IP only (never 0.0.0.0/0).
- Inbound **5001/8501**: your IP only while testing. Both services are
  **unauthenticated** — do not expose them to 0.0.0.0/0 publicly without a
  reverse proxy + auth (see [security.md](security.md)).
- Outbound: default (all) is fine; needed for apt/pip during build only.
- `docker run -p 127.0.0.1:5001:5001` + a reverse proxy is the safer binding.

## 9. How to verify deployment

```bash
docker ps                                  # both containers Up
curl -s http://127.0.0.1:5001/health       # {"status": "ok", ...}
curl -s http://127.0.0.1:5001/model | head
curl -s -X POST http://127.0.0.1:5001/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"year": 2019, "month_num": 9, "quarter": 3,
        "avg_price": 380.55, "list_price": 400.0,
        "sales_lag_1": 15234.11, "sales_lag_2": 14980.44,
        "sales_lag_3": 16001.90, "sales_rolling_3": 15405.48}}'
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8501/   # 200
docker logs sales-api                       # no tracebacks
```

From outside: `http://<EC2-PUBLIC-IP>:8501` in a browser and the same
`curl` commands against `http://<EC2-PUBLIC-IP>:5001`.

## 10. How to stop the service

```bash
docker stop sales-api sales-dashboard   # stop (containers kept)
docker rm sales-api sales-dashboard     # remove
docker compose down                     # if using compose
sudo systemctl stop docker              # stop the daemon itself
```

To avoid charges entirely: **Terminate the instance** in the EC2 console
after saving anything you need (the checkpoint and results are in git).

## 11. Cost & security precautions

- **Cost:** an `t3.small` running 24/7 is the main expense; `t3.micro` is
  free-tier eligible. Always **terminate** idle instances — a stopped
  instance still pays for EBS storage. ECR is optional (local `docker save`
  / `docker load` avoids registry costs).
- **Security:**
  - no secrets exist in this repo or image — keep it that way,
  - SSH key only, restricted security-group source IP,
  - both services are unauthenticated — keep ports private or add a
    reverse proxy with TLS + auth before any public exposure,
  - `docker logs`/CloudWatch for monitoring; keep the AMI patched
    (`apt-get upgrade`), and rebuild images to pick up dependency fixes.
