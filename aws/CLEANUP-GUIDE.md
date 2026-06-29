# AWS Infrastructure Cleanup Guide

## Overview

All infrastructure created for GreenGrid Exchange can be deleted **directly from CloudFormation** - no scripts needed.

**Deletion Time:** 5-10 minutes  
**Result:** All resources deleted, charges stop immediately

---

## Method 1 — Delete via AWS Console (Easiest)

### Step 1 — Go to CloudFormation

1. Open AWS Console: https://console.aws.amazon.com/cloudformation/
2. Make sure region is **us-east-1**
3. Look for stack: **greengrid-rag-ai-poc**

### Step 2 — Delete the Stack

1. Click the stack name: **greengrid-rag-ai-poc**
2. Click the **"Delete"** button (top right)
3. Confirm by clicking **"Delete stack"**

**That's it!** CloudFormation will now delete all resources.

### Step 3 — Monitor Deletion

1. Stay on the stack page
2. Status changes: `DELETE_IN_PROGRESS` → `DELETE_COMPLETE`
3. Refreshes every 30 seconds
4. Takes 5-10 minutes

**Status:** `DELETE_COMPLETE` = All resources deleted ✓

---

## Method 2 — Delete via AWS CLI

### Delete the Stack

```powershell
# PowerShell
aws cloudformation delete-stack `
  --stack-name greengrid-rag-ai-poc `
  --region us-east-1
```

```bash
# Bash/Linux/Mac
aws cloudformation delete-stack \
  --stack-name greengrid-rag-ai-poc \
  --region us-east-1
```

### Wait for Completion

```powershell
# PowerShell
aws cloudformation wait stack-delete-complete `
  --stack-name greengrid-rag-ai-poc `
  --region us-east-1
```

```bash
# Bash/Linux/Mac
aws cloudformation wait stack-delete-complete \
  --stack-name greengrid-rag-ai-poc \
  --region us-east-1
```

### Check Status

```powershell
aws cloudformation describe-stacks `
  --stack-name greengrid-rag-ai-poc `
  --region us-east-1 `
  --query 'Stacks[0].StackStatus'
```

---

## What Gets Deleted

CloudFormation automatically removes everything:

```
✓ EC2 instance          (app stops running)
✓ OpenSearch domain     (all documents/embeddings lost)
✓ S3 bucket             (empty only; objects deleted separately)
✓ VPC & Networking      (subnets, route tables, security groups)
✓ IAM Roles             (permissions cleaned up)
✓ CloudWatch Alarms     (budget alerts removed)
✓ CloudWatch Logs       (3-day retention still applies)
```

**Total:** 20+ resources removed automatically

---

## What Doesn't Get Deleted (Manual Cleanup)

### S3 Bucket Objects

If you uploaded documents to S3:

1. Go to **S3 Console**
2. Open bucket: **greengrid-documents-\<ACCOUNT_ID\>-poc**
3. Select all objects
4. Click **Delete**

### CloudWatch Logs

Logs remain for 3 days after deletion (for debugging):

1. Go to **CloudWatch** → **Log Groups**
2. Find: `/aws/opensearch/greengrid/poc/...`
3. Click the log group
4. Click **"Delete log group"** (optional)

---

## When Do Charges Stop?

| Item | When Stopped |
|---|---|
| EC2 | Immediately (free tier anyway) |
| OpenSearch | ~2 minutes (last charges in that period) |
| S3 | Immediately (if bucket empty) |
| Bedrock | Immediately (no running costs) |
| CloudWatch | Immediately (logs retained 3 days) |

**Result:** Charges essentially stop after deletion completes (5-10 min)

---

## Step-by-Step Walkthrough

### Day 7 Cleanup (2026-07-06)

```
1. Open AWS CloudFormation console
   ↓
2. Click stack: greengrid-rag-ai-poc
   ↓
3. Click "Delete" button
   ↓
4. Confirm "Delete stack"
   ↓
5. Wait for Status: DELETE_COMPLETE (5-10 min)
   ↓
6. Done! ✓ All resources deleted
```

---

## Common Questions

### Q: Will this delete my data?

**A:** Yes. Deleted resources include:
- All documents indexed in OpenSearch
- All embeddings
- S3 objects (if in bucket)

**Recommendation:** Export any important data before deletion.

### Q: Will I be charged after deletion?

**A:** No. Charges stop immediately when resources are deleted (except residual OpenSearch charges for ~2 minutes).

### Q: Can I undo the deletion?

**A:** No. Once deleted, CloudFormation stack is gone. You'd need to:
1. Recreate the stack (re-deploy)
2. Re-upload documents
3. Re-index everything

**Tip:** Keep backups if you need to preserve data.

### Q: What if deletion fails?

**A:** Check CloudFormation Events tab for errors:

1. Click stack name
2. Go to **Events** tab
3. Look for resources with RED status
4. Read the failure reason
5. Common cause: Retained resources (delete manually first)

### Q: How do I know if it's deleted?

**A:** Stack status shows `DELETE_COMPLETE`

Verification:
```powershell
aws cloudformation describe-stacks `
  --stack-name greengrid-rag-ai-poc `
  --region us-east-1
```

If stack doesn't exist, deletion is complete ✓

---

## Before You Delete — Checklist

- [ ] Backup important data from OpenSearch (if needed)
- [ ] Download any documents from S3 (if needed)
- [ ] Export logs or metrics (optional)
- [ ] Verify no other services depend on this stack
- [ ] Set a calendar reminder for Day 7 (2026-07-06)

---

## Quick Reference

| Action | Time | Method |
|---|---|---|
| Delete via Console | 1 click + wait | CloudFormation console |
| Delete via CLI | 1 command + wait | AWS CLI |
| Check status | 30 seconds | Console or CLI |
| Verify deletion | Immediate | Stack shows `DELETE_COMPLETE` |

---

## Resources

| Topic | Link |
|---|---|
| CloudFormation Docs | https://docs.aws.amazon.com/cloudformation/ |
| AWS CLI Reference | https://docs.aws.amazon.com/cli/ |
| Billing FAQ | https://aws.amazon.com/billing/faqs/ |

---

## Summary

**To delete everything:**

1. Go to CloudFormation console
2. Click stack: `greengrid-rag-ai-poc`
3. Click **Delete** button
4. Confirm
5. Wait 5-10 minutes
6. Status shows `DELETE_COMPLETE`
7. Done ✓

**No scripts needed. CloudFormation handles it all.**

---

## Timeline

| Date | Action |
|---|---|
| 2026-06-29 | Stack deployed (today) |
| 2026-06-30 to 2026-07-05 | Test and develop |
| **2026-07-06** | **DELETE STACK** (Day 7 - before charges accumulate) |
| 2026-07-07+ | No more charges ✓ |

---

**Ready to delete?** → Go to AWS CloudFormation console and follow the 3-step delete process!
