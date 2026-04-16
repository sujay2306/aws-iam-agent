import os
import json
import boto3
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def main():
    load_dotenv()

    aws_kwargs = dict(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )

    iam = boto3.client("iam", **aws_kwargs)
    secrets = boto3.client("secretsmanager", **aws_kwargs)

    # ── IAM read tools ───────────────────────────────────────────────

    @tool
    def list_iam_users() -> str:
        """Lists all IAM users in the AWS account."""
        users = iam.list_users()["Users"]
        return json.dumps(
            [{"UserName": u["UserName"], "UserId": u["UserId"], "Arn": u["Arn"]} for u in users],
            indent=2,
        )

    @tool
    def list_iam_roles() -> str:
        """Lists all IAM roles in the AWS account."""
        roles = iam.list_roles()["Roles"]
        return json.dumps([{"RoleName": r["RoleName"], "Arn": r["Arn"]} for r in roles], indent=2)

    @tool
    def list_iam_policies() -> str:
        """Lists all customer-managed IAM policies."""
        policies = iam.list_policies(Scope="Local")["Policies"]
        return json.dumps([{"PolicyName": p["PolicyName"], "Arn": p["Arn"]} for p in policies], indent=2)

    @tool
    def get_user_policies(username: str) -> str:
        """Gets all attached and inline policies for a given IAM user."""
        attached = iam.list_attached_user_policies(UserName=username)["AttachedPolicies"]
        inline = iam.list_user_policies(UserName=username)["PolicyNames"]
        return json.dumps({"attached": attached, "inline": inline}, indent=2)

    @tool
    def get_role_policies(role_name: str) -> str:
        """Gets all attached and inline policies for a given IAM role."""
        attached = iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]
        inline = iam.list_role_policies(RoleName=role_name)["PolicyNames"]
        return json.dumps({"attached": attached, "inline": inline}, indent=2)

    @tool
    def get_policy_document(policy_arn: str) -> str:
        """Gets the policy document (permissions) for a given IAM policy ARN."""
        version_id = iam.get_policy(PolicyArn=policy_arn)["Policy"]["DefaultVersionId"]
        doc = iam.get_policy_version(PolicyArn=policy_arn, VersionId=version_id)["PolicyVersion"]["Document"]
        return json.dumps(doc, indent=2)

    @tool
    def list_access_keys(username: str) -> str:
        """Lists access keys for a given IAM user including status, creation date, and age in days."""
        keys = iam.list_access_keys(UserName=username)["AccessKeyMetadata"]
        now = datetime.now(timezone.utc)
        results = []
        for k in keys:
            age = (now - k["CreateDate"]).days
            results.append({
                "AccessKeyId": k["AccessKeyId"],
                "Status": k["Status"],
                "CreateDate": str(k["CreateDate"]),
                "AgeDays": age,
                "NeedsRotation": age > 90,
            })
        return json.dumps(results, indent=2)

    # ── IAM key rotation tools ───────────────────────────────────────

    @tool
    def create_access_key(username: str) -> str:
        """Creates a new access key pair for an IAM user. Returns the AccessKeyId and SecretAccessKey.
        IMPORTANT: The SecretAccessKey is only available at creation time."""
        response = iam.create_access_key(UserName=username)
        key = response["AccessKey"]
        return json.dumps({
            "AccessKeyId": key["AccessKeyId"],
            "SecretAccessKey": key["SecretAccessKey"],
            "UserName": key["UserName"],
            "Status": key["Status"],
            "CreateDate": str(key["CreateDate"]),
        }, indent=2)

    @tool
    def deactivate_access_key(username: str, access_key_id: str) -> str:
        """Deactivates (but does not delete) an IAM access key. Use this before deleting to safely phase out a key."""
        iam.update_access_key(UserName=username, AccessKeyId=access_key_id, Status="Inactive")
        return json.dumps({"status": "deactivated", "AccessKeyId": access_key_id, "UserName": username})

    @tool
    def delete_access_key(username: str, access_key_id: str) -> str:
        """Permanently deletes an IAM access key. This cannot be undone. Only delete keys that are already inactive."""
        iam.delete_access_key(UserName=username, AccessKeyId=access_key_id)
        return json.dumps({"status": "deleted", "AccessKeyId": access_key_id, "UserName": username})

    # ── Secrets Manager tools ────────────────────────────────────────

    @tool
    def store_secret(secret_name: str, secret_value: str, description: str = "") -> str:
        """Stores or updates a secret in AWS Secrets Manager.
        If the secret already exists it is updated; otherwise a new secret is created.
        Use this to safely store rotated IAM credentials."""
        try:
            secrets.put_secret_value(SecretId=secret_name, SecretString=secret_value)
            action = "updated"
        except secrets.exceptions.ResourceNotFoundException:
            secrets.create_secret(
                Name=secret_name,
                Description=description or f"Managed by IAM Agent",
                SecretString=secret_value,
            )
            action = "created"
        return json.dumps({"status": action, "SecretName": secret_name})

    @tool
    def get_secret(secret_name: str) -> str:
        """Retrieves a secret value from AWS Secrets Manager by name."""
        response = secrets.get_secret_value(SecretId=secret_name)
        return json.dumps({"SecretName": response["Name"], "SecretString": response["SecretString"]}, indent=2)

    @tool
    def list_secrets() -> str:
        """Lists all secrets stored in AWS Secrets Manager (names and ARNs only, not values)."""
        paginator = secrets.get_paginator("list_secrets")
        results = []
        for page in paginator.paginate():
            for s in page["SecretList"]:
                results.append({
                    "Name": s["Name"],
                    "ARN": s["ARN"],
                    "LastChangedDate": str(s.get("LastChangedDate", "")),
                })
        return json.dumps(results, indent=2)

    @tool
    def delete_secret(secret_name: str, force: bool = False) -> str:
        """Deletes a secret from AWS Secrets Manager. By default uses a 30-day recovery window.
        Set force=True to delete immediately without recovery."""
        kwargs = {"SecretId": secret_name}
        if force:
            kwargs["ForceDeleteWithoutRecovery"] = True
        else:
            kwargs["RecoveryWindowInDays"] = 30
        secrets.delete_secret(**kwargs)
        return json.dumps({"status": "scheduled_for_deletion" if not force else "deleted", "SecretName": secret_name})

    @tool
    def rotate_and_store_key(username: str, secret_name: str) -> str:
        """Full key rotation workflow for an IAM user:
        1. Creates a new access key pair
        2. Stores the new credentials in Secrets Manager under the given secret_name
        3. Deactivates all old access keys for the user
        Returns a summary of actions taken."""
        new_key = iam.create_access_key(UserName=username)["AccessKey"]

        secret_payload = json.dumps({
            "AWS_ACCESS_KEY_ID": new_key["AccessKeyId"],
            "AWS_SECRET_ACCESS_KEY": new_key["SecretAccessKey"],
            "UserName": username,
            "RotatedAt": datetime.now(timezone.utc).isoformat(),
        })

        try:
            secrets.put_secret_value(SecretId=secret_name, SecretString=secret_payload)
            secret_action = "updated"
        except secrets.exceptions.ResourceNotFoundException:
            secrets.create_secret(
                Name=secret_name,
                Description=f"Rotated IAM credentials for {username}",
                SecretString=secret_payload,
            )
            secret_action = "created"

        old_keys = iam.list_access_keys(UserName=username)["AccessKeyMetadata"]
        deactivated = []
        for k in old_keys:
            if k["AccessKeyId"] != new_key["AccessKeyId"] and k["Status"] == "Active":
                iam.update_access_key(UserName=username, AccessKeyId=k["AccessKeyId"], Status="Inactive")
                deactivated.append(k["AccessKeyId"])

        return json.dumps({
            "status": "rotation_complete",
            "new_key_id": new_key["AccessKeyId"],
            "secret_name": secret_name,
            "secret_action": secret_action,
            "deactivated_old_keys": deactivated,
        }, indent=2)

    # ── Agent setup ──────────────────────────────────────────────────

    tools = [
        list_iam_users, list_iam_roles, list_iam_policies,
        get_user_policies, get_role_policies, get_policy_document,
        list_access_keys,
        create_access_key, deactivate_access_key, delete_access_key,
        store_secret, get_secret, list_secrets, delete_secret,
        rotate_and_store_key,
    ]

    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced AWS IAM security agent powered by GPT-4.

You can **read** IAM state (users, roles, policies, access keys) and **take action**
(rotate keys, manage secrets). Always use tools to fetch live data before answering.

CAPABILITIES:
- List and inspect IAM users, roles, policies, and access keys
- Detect stale access keys (>90 days old) and recommend rotation
- Rotate IAM access keys: create new key → store in Secrets Manager → deactivate old key
- Store, retrieve, list, and delete secrets in AWS Secrets Manager

SAFETY RULES:
- Before any destructive action (delete key, delete secret) confirm with the user first.
- When rotating keys, always store the new credentials in Secrets Manager BEFORE
  deactivating old keys so credentials are never lost.
- Never display SecretAccessKey values in full — show only the first 4 and last 4 characters.
- If the user asks to rotate keys, prefer `rotate_and_store_key` for an atomic workflow.

Be concise, security-conscious, and proactive about flagging stale keys or risky configs."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

    print("🔐 IAM Agent (GPT-4) ready. Ask anything about your AWS IAM setup.")
    print("   Now with key rotation & Secrets Manager support.")
    print("   Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.strip().lower() in ("exit", "quit"):
            break
        result = agent_executor.invoke({"input": user_input})
        print(f"\nAgent: {result['output']}\n")


if __name__ == "__main__":
    main()
