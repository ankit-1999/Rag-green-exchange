# CloudFormation Template — Complete Architecture Reference

## Overview

This CloudFormation template provisions a complete, cost-optimized infrastructure for the GreenGrid Exchange RAG AI document ingestion pipeline on AWS. Everything is defined-as-code for reproducibility and easy cleanup.

**Template File:** `rag-cloud-formation.yaml`  
**Duration:** 10-15 minutes to create  
**Cost:** ~$70 for 7 days (free tier + $60 credit)

---

## What Gets Created (Complete Resource List)

```
AWS Account (1)
├── VPC (1)
│   ├── Public Subnet (1)
│   ├── Internet Gateway (1)
│   ├── Route Table (1)
│   └── Security Groups (2)
│
├── EC2 Instance (1)
│   └── t3.micro (free tier)
│
├── S3 Bucket (1)
│   └── greengrid-documents-<ACCOUNT_ID>-poc
│
├── OpenSearch Domain (1)
│   └── t3.small.search, 10GB storage
│
├── IAM Roles & Policies (1 role + 4 policies)
│   ├── greengrid-ec2-role
│   ├── S3 permissions
│   ├── Bedrock permissions
│   ├── OpenSearch permissions
│   └── CloudWatch permissions
│
├── CloudWatch Resources (5)
│   ├── 3 budget alarms
│   └── 2 log groups
│
└── Tags on all resources
    └── ExpiresOn: 2026-07-06 (for cleanup)
```

**Total Resources:** 20+

---

## Section-by-Section Breakdown

### 1. Parameters Section

```yaml
Parameters:
  Environment:
    Type: String
    Default: poc
    Description: Environment name
```

**Why:** Allows customization without editing template. Used in resource naming.

**Properties:**
- `Type: String` — User input is text
- `Default: poc` — If not specified, uses "poc"
- Used in: VPC names, EC2 tags, OpenSearch domain name

**Example usage:**
- Environment=poc → `greengrid-vpc-poc`, `greengrid-docs-poc`
- Environment=prod → `greengrid-vpc-prod`, `greengrid-docs-prod`

```yaml
  POCExpirationDate:
    Type: String
    Default: '2026-07-06'
    Description: Date when resources should be cleaned up
```

**Why:** Reminds you when to delete resources to stop incurring charges. Stored as tag on all resources.

**Format:** YYYY-MM-DD (ISO 8601)

---

### 2. VPC & Networking

#### VPC Creation

```yaml
GreenGridVPC:
  Type: AWS::EC2::VPC
  Properties:
    CidrBlock: 10.0.0.0/16
    EnableDnsHostnames: true
    EnableDnsSupport: true
```

**Why:** Isolated network for all resources. Provides private IP address space.

**Properties explained:**
- `CidrBlock: 10.0.0.0/16` — Network range (65,536 IP addresses available)
  - This is a private IP range (RFC 1918), safe for internal use
  - `/16` means first 16 bits define network, last 16 bits for hosts
- `EnableDnsHostnames: true` — EC2 instances get DNS names
- `EnableDnsSupport: true` — VPC can resolve DNS queries

**Cost:** Free

#### Public Subnet

```yaml
PublicSubnet:
  Type: AWS::EC2::Subnet
  Properties:
    VpcId: !Ref GreenGridVPC
    CidrBlock: 10.0.1.0/24
    AvailabilityZone: !Select [0, !GetAZs '']
    MapPublicIpOnLaunch: true
```

**Why:** Sub-network where EC2 will be placed. "Public" means it has internet access.

**Properties explained:**
- `VpcId: !Ref GreenGridVPC` — Attach to VPC we just created (`!Ref` = reference)
- `CidrBlock: 10.0.1.0/24` — Subnet range (256 IPs: 10.0.1.0 to 10.0.1.255)
  - `/24` means first 24 bits define subnet, 8 bits for hosts
  - Part of the larger VPC range (10.0.0.0/16)
- `AvailabilityZone: !Select [0, !GetAZs '']` — Pick first AZ in the region
  - `GetAZs ''` returns all AZs available in region
  - `Select [0, ...]` picks the first one
  - Example: us-east-1a, us-east-1b, us-east-1c → picks us-east-1a
- `MapPublicIpOnLaunch: true` — EC2 instances get public IP automatically

**Cost:** Free

#### Internet Gateway

```yaml
InternetGateway:
  Type: AWS::EC2::InternetGateway
  Properties:
    Tags:
      - Key: Name
        Value: !Sub 'greengrid-igw-${Environment}'

AttachGateway:
  Type: AWS::EC2::VPCGatewayAttachment
  Properties:
    VpcId: !Ref GreenGridVPC
    InternetGatewayId: !Ref InternetGateway
```

**Why:** Gateway that routes traffic between VPC and the internet (0.0.0.0/0).

**Properties explained:**
- `InternetGateway` — The gateway resource itself
- `VPCGatewayAttachment` — Connects gateway to VPC
- `!Sub 'greengrid-igw-${Environment}'` — Template substitution
  - Replaces `${Environment}` with parameter value
  - Example: "greengrid-igw-poc"

**Cost:** Free

#### Route Table & Routes

```yaml
PublicRouteTable:
  Type: AWS::EC2::RouteTable
  Properties:
    VpcId: !Ref GreenGridVPC

PublicRoute:
  Type: AWS::EC2::Route
  DependsOn: AttachGateway
  Properties:
    RouteTableId: !Ref PublicRouteTable
    DestinationCidrBlock: 0.0.0.0/0
    GatewayId: !Ref InternetGateway
```

**Why:** Routing table tells packets where to go. Route says "if destination is anywhere (0.0.0.0/0), send it through Internet Gateway."

**Properties explained:**
- `DestinationCidrBlock: 0.0.0.0/0` — Any IP address (all internet traffic)
- `GatewayId: !Ref InternetGateway` — Send through IGW
- `DependsOn: AttachGateway` — Wait for gateway to be attached first

**Cost:** Free

#### Security Groups

```yaml
EC2SecurityGroup:
  Type: AWS::EC2::SecurityGroup
  Properties:
    GroupDescription: Security group for GreenGrid RAG AI EC2 instance
    VpcId: !Ref GreenGridVPC
    SecurityGroupIngress:
      - IpProtocol: tcp
        FromPort: 22
        ToPort: 22
        CidrIp: 0.0.0.0/0
        Description: SSH access
      - IpProtocol: tcp
        FromPort: 8000
        ToPort: 8000
        CidrIp: 0.0.0.0/0
        Description: FastAPI application
    SecurityGroupEgress:
      - IpProtocol: -1
        CidrIp: 0.0.0.0/0
        Description: Allow all outbound
```

**Why:** Firewall rules. Controls who can connect to EC2.

**Properties explained:**
- `SecurityGroupIngress` — **Inbound rules** (from internet to EC2)
  - Port 22 (SSH) — for you to SSH into server
  - Port 8000 (API) — for API access
  - `0.0.0.0/0` — Anyone on internet can connect
- `SecurityGroupEgress` — **Outbound rules** (from EC2 to internet)
  - `IpProtocol: -1` — All protocols
  - `0.0.0.0/0` — Can reach any destination
- Why outbound? EC2 needs to call S3, Bedrock, OpenSearch

**OpenSearch Security Group:**

```yaml
OpenSearchSecurityGroup:
  Type: AWS::EC2::SecurityGroup
  Properties:
    SecurityGroupIngress:
      - IpProtocol: tcp
        FromPort: 443
        ToPort: 443
        SourceSecurityGroupId: !Ref EC2SecurityGroup
        Description: HTTPS from EC2
```

**Why:** OpenSearch is in a private VPC, only EC2 can access it.

**Properties explained:**
- `SourceSecurityGroupId: !Ref EC2SecurityGroup` — Only EC2 can connect
- Port 443 — HTTPS (encrypted)
- **NOT** open to internet (no `0.0.0.0/0`)

**Cost:** Free

---

### 3. IAM Roles & Policies

#### EC2 Instance Role

```yaml
GreenGridEC2Role:
  Type: AWS::IAM::Role
  Properties:
    RoleName: !Sub 'greengrid-ec2-role-${Environment}'
    AssumeRolePolicyDocument:
      Version: '2012-10-17'
      Statement:
        - Effect: Allow
          Principal:
            Service: ec2.amazonaws.com
          Action: sts:AssumeRole
```

**Why:** Defines what the EC2 instance can do. EC2 will use this role automatically.

**Properties explained:**
- `AssumeRolePolicyDocument` — Trust policy. Says "EC2 service can assume this role"
- `Principal: Service: ec2.amazonaws.com` — Only EC2 can use this role
- This is the trust relationship

#### S3 Permissions

```yaml
S3BucketPolicy:
  Type: AWS::IAM::Policy
  Properties:
    PolicyName: greengrid-s3-policy
    PolicyDocument:
      Version: '2012-10-17'
      Statement:
        - Effect: Allow
          Action:
            - s3:GetObject
            - s3:ListBucket
          Resource:
            - !GetAtt DocumentBucket.Arn
            - !Sub '${DocumentBucket.Arn}/*'
    Roles:
      - !Ref GreenGridEC2Role
```

**Why:** Grants EC2 permission to read documents from S3.

**Properties explained:**
- `s3:GetObject` — Can read objects
- `s3:ListBucket` — Can list bucket contents
- `Resource` — Applied to which buckets/objects
  - `!GetAtt DocumentBucket.Arn` — The bucket itself (ARN = Amazon Resource Name)
  - `${DocumentBucket.Arn}/*` — All objects inside bucket
- `Roles: [GreenGridEC2Role]` — Attach to EC2 role

**Security principle:** Least privilege. Only allows GetObject and ListBucket, nothing else.

**Cost:** Free (IAM is free)

#### Bedrock Permissions

```yaml
BedrockPolicy:
  Type: AWS::IAM::Policy
  Properties:
    PolicyName: greengrid-bedrock-policy
    PolicyDocument:
      Statement:
        - Effect: Allow
          Action:
            - bedrock:InvokeModel
          Resource:
            - !Sub 'arn:aws:bedrock:${AWS::Region}::foundation-model/amazon.titan-embed-text-v2:0'
            - !Sub 'arn:aws:bedrock:${AWS::Region}::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0'
```

**Why:** Grants EC2 permission to call Bedrock for embeddings and text generation.

**Properties explained:**
- `bedrock:InvokeModel` — Can call LLMs and embedding models
- `Resource` specifies exact model ARNs (cannot be too general)
- `${AWS::Region}` — Automatically fills in region (us-east-1, etc.)

**Cost:** You pay per Bedrock API call (not the permission itself)

#### OpenSearch Permissions

```yaml
OpenSearchPolicy:
  Type: AWS::IAM::Policy
  Properties:
    PolicyName: greengrid-opensearch-policy
    PolicyDocument:
      Statement:
        - Effect: Allow
          Action:
            - es:ESHttpGet
            - es:ESHttpHead
            - es:ESHttpPost
            - es:ESHttpPut
            - es:ESHttpDelete
          Resource:
            - !Sub 'arn:aws:es:${AWS::Region}:${AWS::AccountId}:domain/greengrid-docs-${Environment}/*'
```

**Why:** Grants EC2 permission to read/write/delete in OpenSearch.

**Properties explained:**
- `es:ESHttp*` — HTTP methods (GET, HEAD, POST, PUT, DELETE)
- `domain/greengrid-docs-${Environment}/*` — Applied to specific domain + all paths

**Cost:** Free (permission only; you pay for OpenSearch usage)

#### CloudWatch Permissions

```yaml
CloudWatchLogsPolicy:
  Type: AWS::IAM::Policy
  Properties:
    PolicyName: greengrid-cloudwatch-logs-policy
    PolicyDocument:
      Statement:
        - Effect: Allow
          Action:
            - logs:CreateLogGroup
            - logs:CreateLogStream
            - logs:PutLogEvents
            - logs:DescribeLogStreams
          Resource:
            - !Sub 'arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:/greengrid/*'
```

**Why:** Allows EC2 to write application logs to CloudWatch.

**Properties explained:**
- `logs:CreateLogGroup` — Create new log group (e.g., `/greengrid/api`)
- `logs:CreateLogStream` — Create log stream (e.g., `2026-06-29`)
- `logs:PutLogEvents` — Write log entries
- `log-group:/greengrid/*` — All log groups starting with `/greengrid/`

**Cost:** Free for 5GB/month, then ~$0.50 per GB

---

### 4. S3 Bucket

```yaml
DocumentBucket:
  Type: AWS::S3::Bucket
  Properties:
    BucketName: !Sub 'greengrid-documents-${AWS::AccountId}-${Environment}'
    VersioningConfiguration:
      Status: Enabled
    BucketEncryption:
      ServerSideEncryptionConfiguration:
        - ServerSideEncryptionByDefault:
            SSEAlgorithm: AES256
    PublicAccessBlockConfiguration:
      BlockPublicAcls: true
      BlockPublicPolicy: true
      IgnorePublicAcls: true
      RestrictPublicBuckets: true
```

**Why:** Store raw documents before ingestion.

**Properties explained:**
- `BucketName: !Sub 'greengrid-documents-${AWS::AccountId}-${Environment}'`
  - Includes account ID to ensure uniqueness (bucket names are globally unique)
  - Example: `greengrid-documents-123456789012-poc`
- `VersioningConfiguration: Status: Enabled` — Keep old versions of files
  - Allows recovery if file is overwritten
- `SSEAlgorithm: AES256` — Encrypt at rest with AES-256
- `BlockPublicAcls: true` — No public access (private bucket)
- All buckets should be private by default

**Cost:** ~$0.023 per GB stored per month (free tier: 5 GB)

---

### 5. OpenSearch Domain

```yaml
OpenSearchDomain:
  Type: AWS::OpenSearchServiceDomains::Domain
  DependsOn: OpenSearchServiceLinkedRole
  Properties:
    DomainName: !Sub 'greengrid-docs-${Environment}'
    EngineVersion: OpenSearch_2.13
    NodeType: t3.small.search
    InstanceCount: 1
    EBSOptions:
      EBSEnabled: true
      VolumeType: gp3
      VolumeSize: 10
```

**Why:** Vector database for storing document chunks + embeddings for kNN search.

**Properties explained:**
- `DomainName` — Name of the OpenSearch cluster
- `EngineVersion: OpenSearch_2.13` — Latest stable version
- `NodeType: t3.small.search` — Node instance type
  - t3.small = burstable instance, 2 vCPU, 2 GB RAM
  - Cheaper than larger types (cost-optimized for POC)
- `InstanceCount: 1` — Single node (no redundancy for POC)
  - Production would use 3+ nodes across AZs
- `VolumeSize: 10` — 10 GB storage (for ~1000 document chunks)
- `VolumeType: gp3` — General purpose SSD (faster than gp2)

```yaml
    DomainEndpointType: VPC
    VPCOptions:
      SubnetIds:
        - !Ref PublicSubnet
      SecurityGroupIds:
        - !Ref OpenSearchSecurityGroup
```

**Why:** OpenSearch is in VPC (not internet-accessible).

**Properties explained:**
- `DomainEndpointType: VPC` — Private (not public internet)
- `SubnetIds` — Deploy in our public subnet
- `SecurityGroupIds` — Apply OpenSearch security group

```yaml
    AdvancedSecurityOptions:
      Enabled: true
      InternalUserDatabaseEnabled: true
      MasterUserOptions:
        MasterUserName: admin
        MasterUserPassword: !Sub 'GreenGrid${AWS::AccountId}!'
```

**Why:** Require authentication to access OpenSearch.

**Properties explained:**
- `Enabled: true` — Turn on security
- `InternalUserDatabaseEnabled: true` — Use built-in user database (not Cognito)
- `MasterUserName: admin` — Master user login
- `MasterUserPassword: !Sub 'GreenGrid${AWS::AccountId}!'` — Password (change in production!)

**WARNING:** Never use this password format in production. Use Secrets Manager instead.

```yaml
    LogPublishingOptions:
      ES_APPLICATION_LOGS:
        CloudWatchLogsLogGroupArn: !GetAtt OpenSearchLogGroup.Arn
        Enabled: true
      INDEX_SLOW_LOGS:
        CloudWatchLogsLogGroupArn: !GetAtt OpenSearchIndexSlowLogsGroup.Arn
        Enabled: true
```

**Why:** Debug and monitor OpenSearch via CloudWatch logs.

**Properties explained:**
- `ES_APPLICATION_LOGS` — General application logs (errors, info)
- `INDEX_SLOW_LOGS` — Queries slower than 1 second
- Both sent to CloudWatch log groups for analysis

**Cost:** ~$0.30/hour (~$7/day, ~$50 for 7 days) + storage

---

### 6. CloudWatch Resources

```yaml
OpenSearchLogGroup:
  Type: AWS::Logs::LogGroup
  Properties:
    LogGroupName: !Sub '/aws/opensearch/greengrid/${Environment}/application-logs'
    RetentionInDays: 3
```

**Why:** Store OpenSearch logs, auto-delete after 3 days (cost savings).

**Properties explained:**
- `LogGroupName` — Hierarchical name (`/service/app/type`)
- `RetentionInDays: 3` — Keep for 3 days, then delete
  - Balances debugging vs. storage cost
  - Production might use 30+ days

**Cost:** ~$0.03 per million log entries + storage

#### Budget Alarms

```yaml
BudgetAlarm15:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: !Sub 'greengrid-budget-alert-15-${Environment}'
    AlarmDescription: Alert when estimated charges reach $15
    MetricName: EstimatedCharges
    Namespace: AWS/Billing
    Threshold: 15
    ComparisonOperator: GreaterThanOrEqualToThreshold
```

**Why:** Prevent surprise bills. Alarm triggers when spending hits $15, $30, $50.

**Properties explained:**
- `MetricName: EstimatedCharges` — AWS billing metric
- `Namespace: AWS/Billing` — CloudWatch metric namespace
- `Threshold: 15` — Trigger at $15 spent
- `ComparisonOperator: GreaterThanOrEqualToThreshold` — If spending >= $15, alarm fires
- To receive notifications, connect SNS topic (not included in template)

**Cost:** Free (alarms are free)

---

### 7. EC2 Instance

```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  DependsOn: OpenSearchDomain
  Properties:
    ImageId: !Sub '{{resolve:ssm:/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2}}'
    InstanceType: t3.micro
    IamInstanceProfile: !Ref GreenGridEC2RoleInstanceProfile
    SubnetId: !Ref PublicSubnet
    SecurityGroupIds:
      - !Ref EC2SecurityGroup
```

**Why:** Compute instance running the FastAPI app.

**Properties explained:**
- `ImageId: {{resolve:ssm:...}}` — Latest Amazon Linux 2 AMI
  - `resolve:ssm:` — Fetch from Systems Manager Parameter Store
  - Auto-updates to latest AMI (security patches)
- `InstanceType: t3.micro` — Burstable instance (free tier eligible)
  - 1 vCPU, 1 GB RAM
  - Good for low-traffic apps
- `IamInstanceProfile: !Ref GreenGridEC2RoleInstanceProfile` — Attach IAM role
  - EC2 will assume this role automatically
  - No need for AWS keys in code
- `SubnetId: !Ref PublicSubnet` — Deploy in public subnet
- `SecurityGroupIds: [EC2SecurityGroup]` — Apply firewall rules

**Cost:** FREE (free tier: 750 hours per month per account)

#### UserData (startup script)

```bash
#!/bin/bash
set -e

# Update system
yum update -y
yum install -y git python3 python3-pip

# Create app directory
mkdir -p /opt/greengrid
cd /opt/greengrid

# Clone repository
git clone https://github.com/your-org/greenexchange.git . || echo "Git clone skipped"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
if [ -f Rag-green-exchange/requirements.txt ]; then
  pip install -r Rag-green-exchange/requirements.txt
fi

# Create .env file (auto-configured by CloudFormation)
cat > Rag-green-exchange/.env << 'EOF'
AWS_REGION=${AWS::Region}
OPENSEARCH_ENDPOINT=https://${OpenSearchDomain.DomainEndpoint}
S3_BUCKET_NAME=${DocumentBucket}
...
EOF

# Create systemd service (auto-start on reboot)
cat > /etc/systemd/system/greengrid-api.service << 'EOF'
[Unit]
Description=GreenGrid Exchange RAG AI
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/greengrid/Rag-green-exchange
ExecStart=/opt/greengrid/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable greengrid-api.service
systemctl start greengrid-api.service
```

**Why:** Automate everything on EC2 startup.

**What happens:**
1. Update packages
2. Install Python and Git
3. Clone your repository
4. Create virtual environment
5. Install Python dependencies
6. Generate `.env` file with AWS endpoints
7. Create systemd service (auto-restart on failure)
8. Start the FastAPI app

**Result:** App is running 2-3 minutes after EC2 boots

**Cost:** Free (included in EC2 cost)

---

### 8. Outputs

```yaml
Outputs:
  EC2PublicIP:
    Description: Public IP of the EC2 instance
    Value: !GetAtt EC2Instance.PublicIp
    Export:
      Name: !Sub '${Environment}-EC2-PublicIP'

  ApiEndpoint:
    Description: FastAPI application endpoint
    Value: !Sub 'http://${EC2Instance.PublicDnsName}:8000'

  OpenSearchEndpoint:
    Description: OpenSearch domain endpoint
    Value: !Sub 'https://${OpenSearchDomain.DomainEndpoint}'

  DocumentBucketName:
    Description: S3 bucket for documents
    Value: !Ref DocumentBucket

  SSHCommand:
    Description: SSH command to connect to EC2 instance
    Value: !Sub 'ssh -i your-key.pem ec2-user@${EC2Instance.PublicDnsName}'

  CleanupDate:
    Description: Date when resources should be deleted
    Value: !Ref POCExpirationDate
```

**Why:** Provide important information after stack creation.

**Outputs printed to console after successful creation:**
- EC2 public IP → use to SSH
- API endpoint → call `/health`, `/docs`
- OpenSearch endpoint → goes in `.env`
- S3 bucket name → upload documents here
- SSH command → ready-to-copy
- Cleanup date → reminder when to delete

**Cost:** Free (outputs are metadata only)

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│ AWS Account (us-east-1)                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ VPC: 10.0.0.0/16                                     │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Public Subnet: 10.0.1.0/24                     │  │  │
│  │  │  ┌──────────────────────────────────────────┐  │  │  │
│  │  │  │ EC2 Instance (t3.micro) - FREE           │  │  │  │
│  │  │  │  • FastAPI app                            │  │  │  │
│  │  │  │  • IAM role (S3, Bedrock, OpenSearch)    │  │  │  │
│  │  │  │  • Port 22 (SSH), Port 8000 (API)        │  │  │  │
│  │  │  └──────────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ OpenSearch Domain (t3.small) - $7/day         │  │  │
│  │  │  • 10 GB storage                               │  │  │
│  │  │  • knn vector index                            │  │  │
│  │  │  • Private (only EC2 access)                  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ S3 Bucket (greengrid-documents-*) - $0.023/GB       │  │
│  │  • Private, versioned, encrypted                   │  │
│  │  • Documents uploaded here                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ IAM Role: greengrid-ec2-role-poc                    │  │
│  │  • S3:GetObject, S3:ListBucket                     │  │
│  │  • Bedrock:InvokeModel                              │  │
│  │  • ES:ESHttp*                                       │  │
│  │  • CloudWatch logs                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ CloudWatch                                           │  │
│  │  • 3 Budget alarms ($15, $30, $50)                 │  │
│  │  • 2 Log groups (retention: 3 days)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Internet Gateway → Route → Public IP                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

External Services (not created by this template):
  • Bedrock (on-demand)  ~$2-5/day
  • AWS Billing alarms    Free
```

---

## Cost Breakdown

| Resource | Type | Daily | 7 Days | Free Tier? |
|---|---|---|---|---|
| EC2 t3.micro | Compute | FREE | FREE | Yes |
| OpenSearch t3.small | Database | ~$7 | ~$49 | No |
| S3 | Storage | ~$0.01 | ~$0.07 | 5 GB free |
| Bedrock (est.) | AI | ~$2 | ~$14 | 100K tokens free |
| **Total** | | **~$9** | **~$63** | - |

**Your free tier credit:** $60  
**Expected overage:** ~$3 (worth it for testing)

---

## Tags and Organization

All resources tagged with:
```yaml
Tags:
  - Key: Name
    Value: greengrid-*-${Environment}
  - Key: Project
    Value: GreenGrid
  - Key: ExpiresOn
    Value: 2026-07-06
```

**Why tags matter:**
- Identify resources easily in AWS console
- Enable cost allocation (see which resources cost what)
- Automate cleanup (delete all with ExpiresOn=2026-07-06)
- Track ownership and environment

---

## Security Highlights

1. **No hardcoded AWS keys** — EC2 assumes IAM role
2. **Least privilege IAM** — EC2 can only do what it needs
3. **Private OpenSearch** — Only EC2 can connect
4. **Encrypted S3** — AES-256 at rest
5. **Encrypted OpenSearch** — TLS in transit, encrypted at rest
6. **Security groups** — Firewall rules limit access
7. **VPC** — Isolated network
8. **Public/Private separation** — EC2 is public, OpenSearch is private

---

## Next Steps

1. **Deploy this template** → See [AWS-DEPLOYMENT-STEPS.md](./AWS-DEPLOYMENT-STEPS.md)
2. **Enable Bedrock models** → Request access in console
3. **Test document ingestion** → Upload a PDF to S3
4. **Monitor costs** → CloudWatch alarms will notify you
5. **Delete on Day 7** → CloudFormation delete-stack command

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| Stack creation fails | Invalid parameters | Check region, availability zones, IAM permissions |
| EC2 doesn't boot | UserData error | Check /var/log/cloud-init-output.log on EC2 |
| OpenSearch fails to start | Insufficient EBS | Check AWS service quota for OpenSearch |
| No API response | App not started | SSH in and check `sudo systemctl status greengrid-api` |
| Can't access OpenSearch | Security group | Verify EC2 SG and OpenSearch SG allow HTTPS |

---

**For step-by-step deployment instructions, see: `AWS-DEPLOYMENT-STEPS.md`**
