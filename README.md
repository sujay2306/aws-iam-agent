# aws-iam-security-agent

An AI-powered AWS IAM security agent built with LangChain and GPT-4. Query IAM state, audit security posture, simulate permissions, analyze least privilege, rotate access keys, and manage secrets — all from your terminal in plain English.

## What it does

Instead of writing one-off CLI scripts or digging through the AWS console, just ask:

![Demo](demo.png)

The agent figures out which AWS API calls to make, executes them in real time, and returns a plain English answer.

## Capabilities

**Read IAM state** — query users, roles, policies, and access keys.

**Security audit** — scan all users for stale keys, missing MFA, overly permissive policies, and get a risk-scored report (CRITICAL/HIGH/MEDIUM/LOW).

**Credential report** — generate the AWS-native credential report for a full account health snapshot: password age, MFA status, key age, and last login for every user.

**Policy simulator** — test "can user X do action Y on resource Z?" using the IAM Policy Simulator API without actually making the call. Also test draft policies before attaching them.

**Least privilege analysis** — use Access Advisor to find which services a user or role has permissions for but has never used. Get actionable recommendations for tightening permissions.

**Key rotation** — create new access keys, deactivate/delete old ones, detect stale keys (>90 days). Handles the AWS 2-key limit automatically.

**Secrets Manager** — store rotated credentials safely, retrieve or list secrets, clean up old ones.

**Atomic rotation workflow** — one command to create a new key, store it in Secrets Manager, and deactivate the old key.

## Tools

### IAM Read

| Tool | What it does |
|---|---|
| `list_iam_users()` | All users — UserName, UserId, ARN |
| `list_iam_roles()` | All roles and ARNs |
| `list_iam_policies()` | Customer-managed policies only |
| `get_user_policies(username)` | Attached + inline policies for a user |
| `get_role_policies(role_name)` | Attached + inline policies for a role |
| `get_policy_document(policy_arn)` | Full JSON permission document |
| `list_access_keys(username)` | Key IDs, status, creation date, age, rotation flag |

### Security Audit

| Tool | What it does |
|---|---|
| `security_audit()` | Full account scan: stale keys, missing MFA, overly permissive policies, risk score |
| `generate_credential_report()` | AWS-native credential report: password, MFA, key status for all users |

### Policy Simulator

| Tool | What it does |
|---|---|
| `simulate_policy(arn, actions, resources)` | Test if a user/role can perform specific actions on resources |
| `simulate_custom_policy(policy_json, actions, resources)` | Test a draft policy document before attaching it |

### Least Privilege Analysis

| Tool | What it does |
|---|---|
| `get_unused_permissions(arn)` | Find services allowed but never accessed, with removal recommendations |
| `get_last_accessed_details(arn)` | Quick view of when each service was last accessed |

### Key Rotation

| Tool | What it does |
|---|---|
| `create_access_key(username)` | Create a new access key pair |
| `deactivate_access_key(username, key_id)` | Deactivate a key without deleting |
| `delete_access_key(username, key_id)` | Permanently delete a key |
| `rotate_and_store_key(username, secret_name)` | Full rotation: new key, store in Secrets Manager, deactivate old |

### Secrets Manager

| Tool | What it does |
|---|---|
| `store_secret(name, value)` | Create or update a secret |
| `get_secret(name)` | Retrieve a secret value |
| `list_secrets()` | List all secrets (names and ARNs) |
| `delete_secret(name)` | Delete a secret (with recovery window) |

## Installation

### Via pip

```bash
pip3 install aws-iam-agent
```

### Via Homebrew

```bash
brew tap sujay2306/aws-iam-agent
brew install sujay2306/aws-iam-agent/aws-iam-agent
```

## Setup

Create a `.env` file in your working directory:

```
OPENAI_API_KEY=sk-proj-...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

> **Permissions**: For read-only queries and audits, `IAMReadOnlyAccess` is sufficient.
> For key rotation, add `iam:CreateAccessKey`, `iam:UpdateAccessKey`, `iam:DeleteAccessKey`.
> For Secrets Manager, add `SecretsManagerReadWrite`.
> For the policy simulator, add `iam:SimulatePrincipalPolicy` and `iam:SimulateCustomPolicy`.
> For Access Advisor (least privilege), add `iam:GenerateServiceLastAccessedDetails` and `iam:GetServiceLastAccessedDetails`.

## Usage

```bash
aws-iam-agent
```

```
🔐 IAM Agent (GPT-4) ready. Ask anything about your AWS IAM setup.
   Now with key rotation & Secrets Manager support.

You: audit my account
Agent: Security audit complete. Risk: HIGH (7 issues found)
  CRITICAL: deploy-bot has console access but NO MFA
  CRITICAL: terraform-admin has full admin policy (Action:*, Resource:*)
  HIGH: ci-runner key AKIA...WXYZ is 127 days old — needs rotation
  ...

You: can deploy-bot do s3:GetObject on my production bucket?
Agent: Testing with IAM Policy Simulator...
  s3:GetObject → ALLOWED (matched policy: S3ReadAccess)
  deploy-bot CAN read from that S3 bucket.

You: what permissions is sujay-ks not using?
Agent: Access Advisor analysis for sujay-ks:
  42 services allowed, only 6 ever used.
  36 services have NEVER been accessed — candidates for removal:
  - AWS CloudFormation, AWS CodeDeploy, Amazon DynamoDB, ...

You: rotate deploy-bot's key and store it as prod/deploy-bot-creds
Agent: Rotation complete:
  - New key AKIA...ABCD created
  - Stored in Secrets Manager as "prod/deploy-bot-creds"
  - Old key AKIA...WXYZ deactivated

You: exit
```

## Requirements

- Python 3.11+
- OpenAI API key
- AWS credentials with appropriate IAM and Secrets Manager permissions

## Built with

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI GPT-4](https://openai.com)
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

## License

MIT
