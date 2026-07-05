# EC2 Application Operations Guide

Quick reference for starting, stopping, restarting, and checking status of the
GreenGrid RAG API running on EC2.

---

## Prerequisites

You need terminal access to your EC2 instance. Choose one of the two options below:

### Option A — EC2 Instance Connect (Browser Terminal, No Setup Needed)

1. Open [AWS Console](https://console.aws.amazon.com/ec2/)
2. Go to **EC2 → Instances**
3. Select your instance (status must be `running`)
4. Click **Connect** (top right)
5. Choose the **EC2 Instance Connect** tab
6. Click **Connect** — a browser terminal opens
7. Paste and run commands directly in that terminal

> This is the easiest option — no key file or SSH client needed.

### Option B — SSH from Local Machine

```bash
ssh -i /path/to/your-key.pem ec2-user@<EC2_PUBLIC_IP>
```
Replace `<EC2_PUBLIC_IP>` with the IP shown in EC2 Console → Instances → Public IPv4 address.

---

## 1. Start the Application

Run these commands in the EC2 Instance Connect browser terminal (or via SSH):

```bash
# Enable the service so it auto-starts on reboot
sudo systemctl enable greengrid-api

# Start the service now
sudo systemctl start greengrid-api
```

**What each command does:**
- `systemctl enable` — Registers the service with Linux boot system (systemd) so it starts
  automatically whenever the EC2 instance reboots. Run once; stays active.
- `systemctl start` — Actually starts the FastAPI application process right now, without
  needing a reboot.

**Verify it started successfully:**
```bash
# Check service is running (look for "Active: active (running)")
sudo systemctl status greengrid-api --no-pager

# Watch live logs as they appear (Ctrl+C to exit)
sudo journalctl -u greengrid-api -f
```

**Test the API is responding:**
```bash
curl http://localhost:8000/health
```
Expected response: `{"status": "healthy"}` or similar.

---

## 2. Stop the Application

```bash
# Stop the service immediately
sudo systemctl stop greengrid-api

# Prevent it from restarting on next reboot
sudo systemctl disable greengrid-api
```

**What each command does:**
- `systemctl stop` — Immediately terminates the FastAPI process. The EC2 instance stays
  running; only the app is stopped. No data is lost.
- `systemctl disable` — Removes the service from the Linux boot sequence so it won't
  auto-start on the next reboot. Useful when you want to pause the app long-term.

> **Note:** Stopping the app alone does NOT stop EC2 billing. The instance is still
> running and being charged. To stop compute cost, also stop the EC2 instance
> (see Section 4 below).

---

## 3. Restart the Application

Use this after editing `.env` or pulling new code:

```bash
# Reload systemd config if service file changed
sudo systemctl daemon-reload

# Restart the app (stop + start in one command)
sudo systemctl restart greengrid-api

# Confirm it restarted cleanly
sudo systemctl status greengrid-api --no-pager
```

**What each command does:**
- `systemctl daemon-reload` — Re-reads the service unit file at
  `/etc/systemd/system/greengrid-api.service`. Only needed if you edited the service
  file itself (not the `.env`). Safe to run always.
- `systemctl restart` — Stops the current process and starts a fresh one. Picks up any
  environment variable changes in `/opt/greengrid/.env`.

**Check for startup errors after restart:**
```bash
sudo journalctl -u greengrid-api -n 100 --no-pager
```

---

## 4. Stop / Start the EC2 Instance (Pause Billing)

Stopping the EC2 **instance** (not just the app) pauses compute billing.

### Via AWS CLI

**Stop instance:**
```bash
aws ec2 stop-instances \
  --instance-ids i-xxxxxxxxxxxxxxxxx \
  --region us-east-1
```

**Start instance again:**
```bash
aws ec2 start-instances \
  --instance-ids i-xxxxxxxxxxxxxxxxx \
  --region us-east-1
```

Replace `i-xxxxxxxxxxxxxxxxx` with your actual Instance ID (found in EC2 Console → Instances).

### Via AWS Console

**To stop:**
1. Go to EC2 → Instances
2. Select your instance
3. Click **Instance state** → **Stop instance**

**To start:**
1. Go to EC2 → Instances
2. Select your instance (status: `stopped`)
3. Click **Instance state** → **Start instance**
4. Wait ~1 minute for status to become `running`
5. Note the new **Public IPv4 address** (it changes on each start unless you use Elastic IP)

**After starting EC2, start the app:**

1. Go to EC2 → Instances → select your instance → click **Connect** → **EC2 Instance Connect** → **Connect**
2. In the browser terminal, run:

```bash
# Start the app (if auto-start was disabled)
sudo systemctl start greengrid-api
sudo systemctl status greengrid-api --no-pager
```

Or via SSH (use the new Public IPv4 address shown in EC2 console — it changes on each start):
```bash
ssh -i /path/to/your-key.pem ec2-user@<NEW_EC2_PUBLIC_IP>
sudo systemctl start greengrid-api
sudo systemctl status greengrid-api --no-pager
```

---

## 5. Update Environment Variables on EC2

If you change a model ID, endpoint, or any setting:

```bash
# View current .env
cat /opt/greengrid/.env

# Edit .env (nano editor, Ctrl+O to save, Ctrl+X to exit)
sudo nano /opt/greengrid/.env

# Restart app to apply changes
sudo systemctl restart greengrid-api

# Verify change was picked up in logs
sudo journalctl -u greengrid-api -n 50 --no-pager
```

**Example: Update LLM model to Nova Lite:**
```bash
# Remove old model line and add new one
sudo sed -i '/^BEDROCK_LLM_MODEL_ID=/d' /opt/greengrid/.env
echo 'BEDROCK_LLM_MODEL_ID=amazon.nova-micro-v1:0' | sudo tee -a /opt/greengrid/.env

# Confirm the line was written correctly (should show exactly one line)
grep '^BEDROCK_LLM_MODEL_ID=' /opt/greengrid/.env

# Restart to pick up the change
sudo systemctl restart greengrid-api
```

---

## 6. Quick Test After Starting

```bash
# Health check
curl http://localhost:8000/health

# Upload a document (replace with your real S3 URI)
curl -s -X POST http://localhost:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{"s3_uri": "s3://<your-bucket>/sample.pdf", "document_id": "doc-001"}'

# Query the RAG system
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize key points", "top_k": 3}'
```

---

## 7. Cost Summary

| Action | EC2 Compute Cost | OpenSearch Cost | App Running? |
|--------|-----------------|-----------------|--------------|
| App stopped (`systemctl stop`) | Still billing | Still billing | No |
| EC2 instance stopped | **No charge** | Still billing | No |
| CloudFormation stack deleted | No charge | **No charge** | No |

> **Cheapest pause:** Stop EC2 instance (no compute billing).  
> **Zero cost:** Delete the CloudFormation stack (see `CLEANUP-GUIDE.md`).

---

## 8. Service File Location (Reference)

The systemd service that manages the app lives at:
```
/etc/systemd/system/greengrid-api.service
```

The app itself and `.env` live at:
```
/opt/greengrid/
```

App logs are managed by journald and viewable with:
```bash
sudo journalctl -u greengrid-api --since "1 hour ago" --no-pager
```
