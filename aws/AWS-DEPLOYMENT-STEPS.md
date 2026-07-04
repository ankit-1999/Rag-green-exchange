# AWS CloudFormation Deployment — Step-by-Step Guide

## Overview

This guide walks you through deploying the GreenGrid Exchange RAG AI infrastructure using AWS CloudFormation.

**Duration:** 15-20 minutes (CloudFormation creation takes 10-15 minutes)  
**Cost:** ~$70 for 7 days (you have $60 free tier credit + some overage OK)

---

## Prerequisites Checklist

Before you start, verify you have:

- [ ] AWS Account with free tier access
- [ ] AWS Console access (https://console.aws.amazon.com)
- [ ] EC2 key pair created in your region (for SSH access)
- [ ] This file: `rag-cloud-formation.yaml`
- [ ] Region: **us-east-1** (recommended for free tier)

### Create EC2 Key Pair (if you don't have one)

1. Go to **AWS Console** → **EC2** → **Key Pairs**
2. Click **Create key pair**
3. Name: `greenexchange-poc`
4. File format: `.pem` (Mac/Linux) or `.ppk` (Windows/PuTTY)
5. Click **Create key pair**
6. Save the file somewhere safe (you'll need it for SSH)

---

## Step 1 — Open AWS CloudFormation Console

1. Go to **AWS Console**: https://console.aws.amazon.com
2. Search for **"CloudFormation"** or use direct link:
   https://console.aws.amazon.com/cloudformation/home
3. Make sure region is **us-east-1** (top-right corner)

---

## Step 2 — Create a New Stack

1. Click the blue **"Create stack"** button
2. Choose **"Upload a template file"**
3. Click **"Choose file"**
4. Select `rag-cloud-formation.yaml` from your computer
5. Click **"Next"**

---

## Step 3 — Specify Stack Details

**Stack name:**
```
greengrid-rag-ai-poc
```

**Parameters:**

| Parameter | Value | Explanation |
|---|---|---|
| Environment | `poc` | Use "poc" for this test |
| POCExpirationDate | `2026-07-06` | 7 days from today |

**Settings:**
- Leave all other fields as default
- Click **"Next"**

---

## Step 4 — Configure Stack Options

**General:**
- Leave **Tags** empty (optional)
- Leave **Permissions** as default

**Advanced:**
- Leave **Rollback configuration** as default
- Leave **Notification options** empty

**Important:**
- Do NOT check "Enable termination protection" (you want to delete on Day 7)

**Click "Next"**

---

## Step 5 — Review and Create

**Review page:**

1. Scroll down and verify all settings
2. Look for the checkbox at bottom:
   ```
   [ ] I acknowledge that AWS CloudFormation might create IAM resources 
       with custom names.
   ```
3. **MUST CHECK THIS BOX** (required because template creates IAM roles)
4. Click the blue **"Create stack"** button

**You should see:**
```
Stack creation in progress
Status: CREATE_IN_PROGRESS
```

---

## Step 6 — Wait for Stack Creation

**Estimated time:** 10-15 minutes

**Monitor progress:**

Option A — Watch in console:
1. Stay on CloudFormation page
2. Refresh every 30 seconds
3. Status updates: CREATE_IN_PROGRESS → CREATE_COMPLETE

Option B — Check Events tab:
1. Click the **"Events"** tab
2. Shows each resource as it's created
3. Green checkmarks = successful

**What's being created (in order):**
1. VPC and networking (1-2 min)
2. Security groups (1 min)
3. IAM roles (1 min)
4. S3 bucket (1 min)
5. OpenSearch domain (7-8 min) ← longest part
6. EC2 instance (1-2 min)
7. CloudWatch alarms (1 min)

**If something fails:**
- Check the **Events** tab for error messages
- Common issues:
  - Insufficient capacity → try different region
  - IAM permissions → check your AWS account permissions
  - Quota exceeded → check service quotas
- Delete stack and try again

---

## Step 7 — Get Output Values

Once stack shows **"CREATE_COMPLETE"**:

1. Click the **"Outputs"** tab
2. You'll see a table with keys and values:

| Key | Value | Use |
|---|---|---|
| EC2PublicIP | `54.123.45.67` | SSH into instance |
| ApiEndpoint | `http://54.123.45.67:8000` | Access API |
| ApiDocsEndpoint | `http://54.123.45.67:8000/docs` | Open Swagger UI |
| OpenSearchEndpoint | `https://greengrid-docs-poc-a1b2c3d.us-east-1.es.amazonaws.com` | Goes in `.env` |
| DocumentBucketName | `greengrid-documents-123456789012-poc` | Upload PDFs here |
| SSHCommand | `ssh -i your-key.pem ec2-user@ec2-54-123-45-67.compute-1.amazonaws.com` | SSH into instance |
| CleanupDate | `2026-07-06` | Delete stack this date |

**Copy these values somewhere safe** (paste into a text editor)

---

## Step 8 — Enable Bedrock Models

CloudFormation created the infrastructure, but Bedrock models still need approval.

1. Go to **AWS Console** → search for **"Bedrock"** → open it
2. Make sure region is **us-east-1**
3. Left sidebar: Click **"Model access"**
4. Click the **"Manage model access"** button
5. Search for and enable:
   - [ ] `amazon.titan-embed-text-v2:0`
   - [ ] `anthropic.claude-haiku-4-5-20251001-v1:0`
6. Click **"Save changes"**

**Status:**
- Usually shows "Access granted" immediately (free tier)
- If pending, wait 5-10 minutes

---

## Step 9 — Verify API is Running

EC2 startup script takes 2-3 minutes. Let's check if the app is ready.

### Option A — Health Check (easiest)

Open your browser and go to:
```
http://<EC2PublicIP>:8000/health
```

Replace `<EC2PublicIP>` with the value from Outputs (e.g., `54.123.45.67`)

**Expected response:**
```json
{"status":"ok","service":"greengrid-rag-ai"}
```

If you get **"Connection refused"**, wait another minute and retry (EC2 is still starting up).

### Option B — SSH Into EC2 (if needed)

```bash
# Windows PowerShell
ssh -i C:\path\to\greenexchange-poc.pem ec2-user@<EC2PublicIP>

# Mac/Linux
ssh -i ~/greenexchange-poc.pem ec2-user@<EC2PublicIP>
```

Check app status:
```bash
sudo systemctl status greengrid-api.service
sudo journalctl -u greengrid-api.service -n 20
```

---

## Step 10 — Test the API

Once health check works:

1. Open your browser to:
   ```
   http://<EC2PublicIP>:8000/docs
   ```

2. You should see **Swagger UI** (interactive API documentation)

3. Test the endpoints:
   - **GET /health** → should return `{"status":"ok"}`
   - **GET /** → should return service info
   - **GET /documents** → should return `[]` (no documents yet)

---

## Step 11 — Upload a Test Document

### Create a test file

```bash
# On your local machine, create test.txt with content:
Solar, wind, and hydro are renewable electricity sources.
Coal, gas, and thermal are fossil sources and should not be marked as green.
One electricity credit represents 1 kWh of verified electricity generation.
Retired credits cannot be bought, sold, transferred, listed, or reused.
```

### Upload to S3

1. Go to **AWS Console** → **S3**
2. Find bucket: `greengrid-documents-<ACCOUNT_ID>-poc`
3. Click **Upload**
4. Select `test.txt`
5. Click **Upload**
6. Wait for success message

### Trigger document ingestion

1. Go to **Swagger UI**: `http://<EC2PublicIP>:8000/docs`
2. Find **POST /documents/upload**
3. Click **"Try it out"**
4. Fill in:
   ```json
   {
     "document_name": "test.txt",
     "document_type": "POLICY",
     "s3_uri": "s3://greengrid-documents-<ACCOUNT_ID>-poc/test.txt"
   }
   ```
   (Replace `<ACCOUNT_ID>` with your 12-digit AWS account ID)

5. Click **Execute**

**Expected response (200 OK):**
```json
{
  "document_id": "doc_a3f9bc12",
  "document_name": "test.txt",
  "status": "indexed",
  "chunk_count": 2,
  "message": "Document indexed successfully. 2/2 chunks stored."
}
```

**If you get errors:**
- Check Bedrock models are enabled
- Check S3 bucket name is correct
- Check EC2 instance has internet access
- Check EC2 logs: `sudo journalctl -u greengrid-api.service`

---

## Step 12 — Monitor Costs

### Check current spending

1. Go to **AWS Console** → **Billing**
2. Click **"Cost Management"** → **"Costs by service"**
3. Look for:
   - EC2 (should be FREE)
   - OpenSearch (should show ~$7 per day)
   - S3 (should show ~$0.01)
   - Bedrock (should show charges for API calls you made)

### Budget alarms

1. Go to **AWS Console** → **CloudWatch** → **Alarms**
2. You should see 3 alarms:
   - `greengrid-budget-alert-15-poc` ($15)
   - `greengrid-budget-alert-30-poc` ($30)
   - `greengrid-budget-alert-50-poc` ($50)

If spending exceeds thresholds, they'll trigger. (You need SNS topic to receive emails)

---

## Step 13 — Monitor Infrastructure Health

### EC2 Status

1. Go to **AWS Console** → **EC2** → **Instances**
2. Click **"greengrid-app-poc"** instance
3. Check:
   - **Instance State**: `running`
   - **Status Checks**: `2/2 passed`

### OpenSearch Status

1. Go to **AWS Console** → **OpenSearch** → **Domains**
2. Click **"greengrid-docs-poc"** domain
3. Check:
   - **Domain Status**: `Active`
   - **Nodes**: `1`
   - **Free storage space**: should show available GB

### S3 Bucket

1. Go to **AWS Console** → **S3**
2. Click your bucket
3. Check:
   - Objects listed (documents you uploaded)
   - Encryption: `Enabled`

### CloudWatch Logs

1. Go to **AWS Console** → **CloudWatch** → **Log Groups**
2. Look for:
   - `/aws/opensearch/greengrid/poc/application-logs` (OpenSearch logs)
   - `/aws/opensearch/greengrid/poc/index-slow-logs` (slow queries)

---

## Step 14 — SSH Into EC2 (Optional)

To inspect the app manually:

```bash
ssh -i ~/greenexchange-poc.pem ec2-user@<EC2PublicIP>

# Once connected:
cd /opt/greengrid/Rag-green-exchange

# View app logs
sudo journalctl -u greengrid-api.service -f

# Check if app is running
sudo systemctl status greengrid-api

# View .env file
cat .env

# Exit
exit
```

---

## Step 15 — Cleanup (Day 7)

On July 6, 2026, delete everything to stop charges.

### Via AWS Console

1. Go to **AWS CloudFormation**
2. Click **"greengrid-rag-ai-poc"** stack
3. Click **"Delete"** button
4. Confirm **"Delete stack"**
5. Wait for status: **DELETE_COMPLETE**

**All resources deleted:**
- ✅ EC2 instance
- ✅ OpenSearch domain
- ✅ S3 bucket (empty, if you delete objects first)
- ✅ VPC and networking
- ✅ IAM roles
- ✅ CloudWatch alarms and logs

**Result:** $0/month after this date.

### Via AWS CLI (optional)

```powershell
aws cloudformation delete-stack `
  --stack-name greengrid-rag-ai-poc `
  --region us-east-1

# Wait for completion
aws cloudformation wait stack-delete-complete `
  --stack-name greengrid-rag-ai-poc `
  --region us-east-1
```

---

## Troubleshooting

### Stack creation fails

**Check:**
1. Go to CloudFormation → **Events** tab
2. Look for resources with RED **Status**
3. Read the **Reason** message

**Common causes:**
- Insufficient EC2 capacity → Pick different region
- IAM permission denied → Check your AWS account permissions
- Service quota exceeded → Request quota increase
- Region not supported → Use us-east-1

**Solution:**
1. Delete failed stack
2. Fix the issue
3. Try again

### API not responding

**Check:**
1. Verify EC2 instance is running: EC2 console
2. Check app status: `sudo systemctl status greengrid-api.service`
3. View logs: `sudo journalctl -u greengrid-api.service -n 50`

**Common causes:**
- App hasn't started yet (wait 3 minutes after EC2 boots)
- Dependencies not installed (check pip install output)
- Port 8000 blocked (check security group)

### Document ingestion fails

**Check Bedrock models are enabled:**
1. Go to Bedrock console
2. Click **Model access**
3. Verify both models show **"Access granted"**

**Check S3 bucket access:**
1. S3 URI is correct
2. File exists in bucket
3. IAM role has `s3:GetObject` permission

**Check logs:**
```bash
sudo journalctl -u greengrid-api.service -f
# Upload a document and watch logs in real-time
```

### High costs

**Check spending:**
1. AWS Billing console
2. Look for unexpected charges

**Cost control:**
- Delete stack immediately (don't wait 7 days)
- Disable Bedrock models (stop charges)
- Delete S3 objects (save storage cost)

---

## Next Steps

1. ✅ Stack deployed and running
2. ✅ API responding to requests
3. ✅ Document ingestion tested
4. → **Phase 2:** Build AI/RAG query pipeline (`/ai/ask` endpoint)
5. → **Phase 3:** Connect everything end-to-end
6. → **Phase 4:** Production deployment

---

## Support Resources

| Resource | Link | Use for |
|---|---|---|
| CloudFormation docs | https://docs.aws.amazon.com/cloudformation/ | Template reference |
| EC2 docs | https://docs.aws.amazon.com/ec2/ | Instance management |
| OpenSearch docs | https://docs.aws.amazon.com/opensearch-service/ | Vector database |
| Bedrock docs | https://docs.aws.amazon.com/bedrock/ | LLM integration |
| AWS CLI docs | https://docs.aws.amazon.com/cli/ | Command-line tools |

---

## Cheat Sheet

| Task | Command/Step |
|---|---|
| View stack status | CloudFormation → Stacks → click stack |
| Get outputs | Stacks → Outputs tab |
| Delete stack | Stacks → stack name → Delete button |
| SSH into EC2 | `ssh -i key.pem ec2-user@<IP>` |
| Check app status | `sudo systemctl status greengrid-api` |
| View app logs | `sudo journalctl -u greengrid-api -f` |
| Check spending | Billing → Cost Management |
| Enable Bedrock | Bedrock → Model access → Manage → Save |
| Upload document | S3 console → Upload |
| Test API | Browser → `http://<IP>:8000/docs` |

---

**For detailed explanations of the template resources, see: `TEMPLATE-REFERENCE.md`**
