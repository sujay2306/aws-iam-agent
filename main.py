import os
import io
import csv
import json
import time
import boto3
import functools
from datetime import datetime, timezone
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent


def _safe_tool(func):
    """Wraps a tool function so AWS/boto errors are returned as JSON
    error messages to the LLM instead of crashing the agent."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg = e.response["Error"]["Message"]
            return json.dumps({"error": code, "message": msg})
        except Exception as e:
            return json.dumps({"error": type(e).__name__, "message": str(e)})
    return wrapper


def main():
    load_dotenv()
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        for fallback in [
            os.path.join(os.path.expanduser("~"), ".env"),
            os.path.join(os.path.expanduser("~"), ".aws-iam-agent.env"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        ]:
            if os.path.exists(fallback):
                load_dotenv(fallback, override=True)
                break

    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Missing credentials. Create a .env file in the current directory with:")
        print("  OPENAI_API_KEY=sk-proj-...")
        print("  AWS_ACCESS_KEY_ID=AKIA...")
        print("  AWS_SECRET_ACCESS_KEY=...")
        print("  AWS_REGION=us-east-1")
        print()
        print("Or place it at ~/.env or ~/.aws-iam-agent.env")
        return

    aws_kwargs = dict(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )

    iam = boto3.client("iam", **aws_kwargs)
    secrets = boto3.client("secretsmanager", **aws_kwargs)
    cloudtrail = boto3.client("cloudtrail", **aws_kwargs)

    # ── IAM read tools ───────────────────────────────────────────────

    @tool
    @_safe_tool
    def list_iam_users() -> str:
        """Lists all IAM users in the AWS account."""
        users = iam.list_users()["Users"]
        return json.dumps(
            [{"UserName": u["UserName"], "UserId": u["UserId"], "Arn": u["Arn"]} for u in users],
            indent=2,
        )

    @tool
    @_safe_tool
    def list_iam_roles() -> str:
        """Lists all IAM roles in the AWS account."""
        roles = iam.list_roles()["Roles"]
        return json.dumps([{"RoleName": r["RoleName"], "Arn": r["Arn"]} for r in roles], indent=2)

    @tool
    @_safe_tool
    def list_iam_policies() -> str:
        """Lists all customer-managed IAM policies."""
        policies = iam.list_policies(Scope="Local")["Policies"]
        return json.dumps([{"PolicyName": p["PolicyName"], "Arn": p["Arn"]} for p in policies], indent=2)

    @tool
    @_safe_tool
    def get_user_policies(username: str) -> str:
        """Gets all attached and inline policies for a given IAM user."""
        attached = iam.list_attached_user_policies(UserName=username)["AttachedPolicies"]
        inline = iam.list_user_policies(UserName=username)["PolicyNames"]
        return json.dumps({"attached": attached, "inline": inline}, indent=2)

    @tool
    @_safe_tool
    def get_role_policies(role_name: str) -> str:
        """Gets all attached and inline policies for a given IAM role."""
        attached = iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]
        inline = iam.list_role_policies(RoleName=role_name)["PolicyNames"]
        return json.dumps({"attached": attached, "inline": inline}, indent=2)

    @tool
    @_safe_tool
    def get_policy_document(policy_arn: str) -> str:
        """Gets the policy document (permissions) for a given IAM policy ARN."""
        version_id = iam.get_policy(PolicyArn=policy_arn)["Policy"]["DefaultVersionId"]
        doc = iam.get_policy_version(PolicyArn=policy_arn, VersionId=version_id)["PolicyVersion"]["Document"]
        return json.dumps(doc, indent=2)

    @tool
    @_safe_tool
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
    @_safe_tool
    def create_access_key(username: str) -> str:
        """Creates a new access key pair for an IAM user. Returns the AccessKeyId and SecretAccessKey.
        IMPORTANT: The SecretAccessKey is only available at creation time.
        NOTE: AWS allows max 2 keys per user. If the user already has 2 keys,
        delete one first using delete_access_key before calling this."""
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
    @_safe_tool
    def deactivate_access_key(username: str, access_key_id: str) -> str:
        """Deactivates (but does not delete) an IAM access key. Use this before deleting to safely phase out a key."""
        iam.update_access_key(UserName=username, AccessKeyId=access_key_id, Status="Inactive")
        return json.dumps({"status": "deactivated", "AccessKeyId": access_key_id, "UserName": username})

    @tool
    @_safe_tool
    def delete_access_key(username: str, access_key_id: str) -> str:
        """Permanently deletes an IAM access key. This cannot be undone. Only delete keys that are already inactive."""
        iam.delete_access_key(UserName=username, AccessKeyId=access_key_id)
        return json.dumps({"status": "deleted", "AccessKeyId": access_key_id, "UserName": username})

    # ── Secrets Manager tools ────────────────────────────────────────

    @tool
    @_safe_tool
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
                Description=description or "Managed by IAM Agent",
                SecretString=secret_value,
            )
            action = "created"
        return json.dumps({"status": action, "SecretName": secret_name})

    @tool
    @_safe_tool
    def get_secret(secret_name: str) -> str:
        """Retrieves a secret value from AWS Secrets Manager by name."""
        response = secrets.get_secret_value(SecretId=secret_name)
        return json.dumps({"SecretName": response["Name"], "SecretString": response["SecretString"]}, indent=2)

    @tool
    @_safe_tool
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
    @_safe_tool
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
    @_safe_tool
    def rotate_and_store_key(username: str, secret_name: str) -> str:
        """Full key rotation workflow for an IAM user:
        1. Checks existing keys — if user already has 2, deletes the oldest/inactive one first
        2. Creates a new access key pair
        3. Stores the new credentials in Secrets Manager under the given secret_name
        4. Deactivates all remaining old access keys
        Returns a summary of actions taken.
        AWS allows max 2 access keys per user, so this handles the limit automatically."""
        now = datetime.now(timezone.utc)
        existing_keys = iam.list_access_keys(UserName=username)["AccessKeyMetadata"]

        if not existing_keys:
            return json.dumps({
                "status": "skipped",
                "reason": "no_access_keys",
                "message": f"User '{username}' has no access keys — nothing to rotate.",
                "UserName": username,
            })

        deleted_to_make_room = None
        if len(existing_keys) >= 2:
            oldest = sorted(existing_keys, key=lambda k: k["CreateDate"])[0]
            if oldest["Status"] == "Active":
                iam.update_access_key(
                    UserName=username, AccessKeyId=oldest["AccessKeyId"], Status="Inactive"
                )
            iam.delete_access_key(UserName=username, AccessKeyId=oldest["AccessKeyId"])
            deleted_to_make_room = oldest["AccessKeyId"]
            existing_keys = [k for k in existing_keys if k["AccessKeyId"] != oldest["AccessKeyId"]]

        new_key = iam.create_access_key(UserName=username)["AccessKey"]

        secret_payload = json.dumps({
            "AWS_ACCESS_KEY_ID": new_key["AccessKeyId"],
            "AWS_SECRET_ACCESS_KEY": new_key["SecretAccessKey"],
            "UserName": username,
            "RotatedAt": now.isoformat(),
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

        deactivated = []
        for k in existing_keys:
            if k["AccessKeyId"] != new_key["AccessKeyId"] and k["Status"] == "Active":
                iam.update_access_key(
                    UserName=username, AccessKeyId=k["AccessKeyId"], Status="Inactive"
                )
                deactivated.append(k["AccessKeyId"])

        return json.dumps({
            "status": "rotation_complete",
            "new_key_id": new_key["AccessKeyId"],
            "secret_name": secret_name,
            "secret_action": secret_action,
            "deleted_to_make_room": deleted_to_make_room,
            "deactivated_old_keys": deactivated,
        }, indent=2)

    # ── Security audit tools ─────────────────────────────────────────

    @tool
    @_safe_tool
    def security_audit() -> str:
        """Runs a full IAM security audit across all users. Checks for:
        - Stale access keys (>90 days old)
        - Inactive access keys that should be cleaned up
        - Users without MFA enabled
        - Overly permissive policies (Action:* + Resource:*)
        Returns a structured JSON report with per-user findings and an overall risk score."""
        now = datetime.now(timezone.utc)
        users = iam.list_users()["Users"]
        findings = []
        total_issues = 0

        for user in users:
            uname = user["UserName"]
            user_issues = []

            keys = iam.list_access_keys(UserName=uname)["AccessKeyMetadata"]
            for k in keys:
                age = (now - k["CreateDate"]).days
                if k["Status"] == "Inactive":
                    user_issues.append({"severity": "LOW", "issue": f"Inactive key {k['AccessKeyId']} should be deleted"})
                elif age > 90:
                    user_issues.append({"severity": "HIGH", "issue": f"Key {k['AccessKeyId']} is {age} days old — needs rotation"})
                elif age > 60:
                    user_issues.append({"severity": "MEDIUM", "issue": f"Key {k['AccessKeyId']} is {age} days old — rotation recommended soon"})

            mfa_devices = iam.list_mfa_devices(UserName=uname)["MFADevices"]
            try:
                login_profile = iam.get_login_profile(UserName=uname)
                has_console = True
            except iam.exceptions.NoSuchEntityException:
                has_console = False

            if not mfa_devices:
                if has_console:
                    user_issues.append({"severity": "CRITICAL", "issue": "Console access enabled but NO MFA device"})
                elif keys:
                    user_issues.append({"severity": "MEDIUM", "issue": "No MFA device configured"})

            attached = iam.list_attached_user_policies(UserName=uname)["AttachedPolicies"]
            for pol in attached:
                try:
                    version_id = iam.get_policy(PolicyArn=pol["PolicyArn"])["Policy"]["DefaultVersionId"]
                    doc = iam.get_policy_version(PolicyArn=pol["PolicyArn"], VersionId=version_id)["PolicyVersion"]["Document"]
                    stmts = doc.get("Statement", [])
                    if isinstance(stmts, dict):
                        stmts = [stmts]
                    for stmt in stmts:
                        actions = stmt.get("Action", "")
                        resources = stmt.get("Resource", "")
                        if stmt.get("Effect") == "Allow" and (actions == "*" or actions == ["*"]) and (resources == "*" or resources == ["*"]):
                            user_issues.append({"severity": "CRITICAL", "issue": f"Policy '{pol['PolicyName']}' grants full admin (Action:*, Resource:*)"})
                            break
                except Exception:
                    pass

            total_issues += len(user_issues)
            findings.append({
                "UserName": uname,
                "HasConsoleAccess": has_console,
                "HasMFA": len(mfa_devices) > 0,
                "AccessKeyCount": len(keys),
                "Issues": user_issues,
                "IssueCount": len(user_issues),
            })

        if total_issues == 0:
            risk = "LOW"
        elif total_issues <= 3:
            risk = "MEDIUM"
        elif total_issues <= 8:
            risk = "HIGH"
        else:
            risk = "CRITICAL"

        critical = sum(1 for f in findings for i in f["Issues"] if i["severity"] == "CRITICAL")
        high = sum(1 for f in findings for i in f["Issues"] if i["severity"] == "HIGH")

        return json.dumps({
            "audit_date": now.isoformat(),
            "total_users": len(users),
            "total_issues": total_issues,
            "critical_issues": critical,
            "high_issues": high,
            "overall_risk": risk,
            "users": findings,
        }, indent=2)

    @tool
    @_safe_tool
    def generate_credential_report() -> str:
        """Generates and parses the AWS IAM credential report (account-wide).
        Returns structured data for every user: password age, MFA status,
        access key age, last used dates, and console access.
        This is the most comprehensive single-call account health snapshot."""
        iam.generate_credential_report()
        for _ in range(15):
            time.sleep(2)
            try:
                resp = iam.get_credential_report()
                if resp["ReportFormat"] == "text/csv":
                    break
            except iam.exceptions.CredentialReportNotReadyException:
                continue
        else:
            return json.dumps({"error": "timeout", "message": "Credential report not ready after 30s"})

        content = resp["Content"].decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        results = []
        for row in reader:
            results.append({
                "user": row.get("user", ""),
                "arn": row.get("arn", ""),
                "password_enabled": row.get("password_enabled", ""),
                "password_last_used": row.get("password_last_used", ""),
                "password_last_changed": row.get("password_last_changed", ""),
                "mfa_active": row.get("mfa_active", ""),
                "access_key_1_active": row.get("access_key_1_active", ""),
                "access_key_1_last_rotated": row.get("access_key_1_last_rotated", ""),
                "access_key_1_last_used_date": row.get("access_key_1_last_used_date", ""),
                "access_key_2_active": row.get("access_key_2_active", ""),
                "access_key_2_last_rotated": row.get("access_key_2_last_rotated", ""),
                "access_key_2_last_used_date": row.get("access_key_2_last_used_date", ""),
            })

        return json.dumps({"generated_at": str(resp["GeneratedTime"]), "users": results}, indent=2)

    # ── Policy simulator tools ───────────────────────────────────────

    @tool
    @_safe_tool
    def simulate_policy(principal_arn: str, action_names: str, resource_arns: str = "*") -> str:
        """Tests whether an IAM user or role can perform specific actions on specific resources,
        WITHOUT actually making the API call. Uses the IAM Policy Simulator.

        Args:
            principal_arn: ARN of the user or role to test (e.g. arn:aws:iam::123456789012:user/deploy-bot)
            action_names: Comma-separated AWS actions to test (e.g. "s3:GetObject,s3:PutObject")
            resource_arns: Comma-separated resource ARNs to test against (default "*" for all)

        Returns per-action allow/deny verdict with the matching policy."""
        actions = [a.strip() for a in action_names.split(",")]
        resources = [r.strip() for r in resource_arns.split(",")]

        response = iam.simulate_principal_policy(
            PolicySourceArn=principal_arn,
            ActionNames=actions,
            ResourceArns=resources,
        )

        results = []
        for r in response["EvaluationResults"]:
            results.append({
                "Action": r["EvalActionName"],
                "Resource": r.get("EvalResourceName", "*"),
                "Decision": r["EvalDecision"],
                "MatchedStatements": [
                    {"SourcePolicyId": s.get("SourcePolicyId", ""), "SourcePolicyType": s.get("SourcePolicyType", "")}
                    for s in r.get("MatchedStatements", [])
                ],
            })

        return json.dumps({"principal": principal_arn, "results": results}, indent=2)

    @tool
    @_safe_tool
    def simulate_custom_policy(policy_json: str, action_names: str, resource_arns: str = "*") -> str:
        """Tests a policy document you're drafting BEFORE attaching it.
        Useful for verifying 'would this policy allow action X on resource Y?'

        Args:
            policy_json: The full IAM policy JSON document to test
            action_names: Comma-separated AWS actions to test (e.g. "s3:GetObject,ec2:RunInstances")
            resource_arns: Comma-separated resource ARNs to test against (default "*" for all)

        Returns per-action allow/deny verdict."""
        actions = [a.strip() for a in action_names.split(",")]
        resources = [r.strip() for r in resource_arns.split(",")]

        response = iam.simulate_custom_policy(
            PolicyInputList=[policy_json],
            ActionNames=actions,
            ResourceArns=resources,
        )

        results = []
        for r in response["EvaluationResults"]:
            results.append({
                "Action": r["EvalActionName"],
                "Resource": r.get("EvalResourceName", "*"),
                "Decision": r["EvalDecision"],
            })

        return json.dumps({"results": results}, indent=2)

    # ── Least privilege analysis tools ───────────────────────────────

    @tool
    @_safe_tool
    def get_unused_permissions(principal_arn: str) -> str:
        """Analyzes which AWS services a user or role has permissions for but has NEVER used.
        Calls Access Advisor to compare allowed vs actually-accessed services.

        Args:
            principal_arn: ARN of the IAM user or role to analyze

        Returns: services allowed but never used, services last accessed, and recommendations."""
        job = iam.generate_service_last_accessed_details(Arn=principal_arn)
        job_id = job["JobId"]

        for _ in range(15):
            time.sleep(2)
            details = iam.get_service_last_accessed_details(JobId=job_id)
            if details["JobStatus"] == "COMPLETED":
                break
        else:
            return json.dumps({"error": "timeout", "message": "Access Advisor job not completed after 30s"})

        accessed_services = {}
        never_used = []
        for svc in details["ServicesLastAccessed"]:
            name = svc["ServiceName"]
            ns = svc["ServiceNamespace"]
            last = svc.get("LastAuthenticated")
            if last:
                days_ago = (datetime.now(timezone.utc) - last).days
                accessed_services[ns] = {
                    "ServiceName": name,
                    "LastAccessed": str(last),
                    "DaysAgo": days_ago,
                }
            else:
                never_used.append({"ServiceName": name, "ServiceNamespace": ns})

        return json.dumps({
            "principal": principal_arn,
            "total_services_allowed": len(details["ServicesLastAccessed"]),
            "services_used": len(accessed_services),
            "services_never_used": len(never_used),
            "never_used_services": never_used,
            "accessed_services": accessed_services,
            "recommendation": f"{len(never_used)} service permissions have never been used and are candidates for removal.",
        }, indent=2)

    @tool
    @_safe_tool
    def get_last_accessed_details(principal_arn: str) -> str:
        """Quick check: when was each AWS service last accessed by a user or role?
        Lightweight version of get_unused_permissions — just the access dates, no comparison.

        Args:
            principal_arn: ARN of the IAM user or role to check"""
        job = iam.generate_service_last_accessed_details(Arn=principal_arn)
        job_id = job["JobId"]

        for _ in range(15):
            time.sleep(2)
            details = iam.get_service_last_accessed_details(JobId=job_id)
            if details["JobStatus"] == "COMPLETED":
                break
        else:
            return json.dumps({"error": "timeout", "message": "Access Advisor job not completed after 30s"})

        results = []
        for svc in details["ServicesLastAccessed"]:
            last = svc.get("LastAuthenticated")
            results.append({
                "ServiceName": svc["ServiceName"],
                "ServiceNamespace": svc["ServiceNamespace"],
                "LastAccessed": str(last) if last else "Never",
                "DaysAgo": (datetime.now(timezone.utc) - last).days if last else None,
            })

        results.sort(key=lambda x: x["DaysAgo"] if x["DaysAgo"] is not None else 999999)

        return json.dumps({"principal": principal_arn, "services": results}, indent=2)

    # ── Least privilege policy generation (CloudTrail-based) ────────

    @tool
    @_safe_tool
    def least_privilege_advisor(identity_name: str, identity_type: str = "user", lookback_days: int = 90) -> str:
        """Generates a least-privilege IAM policy based on ACTUAL API usage from CloudTrail event history.
        Looks back through CloudTrail events for the given user or role, collects every unique
        action and resource ARN that was actually used, and builds a minimal IAM policy document
        that grants only those permissions.

        Args:
            identity_name: The IAM user name or role name to analyze
            identity_type: "user" or "role" (default "user")
            lookback_days: How many days of CloudTrail history to scan (default 90, max 90 for free-tier event history)

        Returns: A ready-to-use IAM policy JSON, plus a summary of actions discovered."""
        from datetime import timedelta
        from collections import defaultdict

        lookback_days = min(lookback_days, 90)
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(days=lookback_days)

        if identity_type == "role":
            username_filter = f"arn:aws:sts::{_get_account_id(iam)}:assumed-role/{identity_name}"
        else:
            username_filter = identity_name

        service_action_map = defaultdict(set)
        resource_arns = defaultdict(set)
        event_count = 0
        next_token = None

        while True:
            lookup_kwargs = {
                "LookupAttributes": [
                    {"AttributeKey": "Username", "AttributeValue": username_filter}
                ],
                "StartTime": start_time,
                "EndTime": now,
                "MaxResults": 50,
            }
            if next_token:
                lookup_kwargs["NextToken"] = next_token

            response = cloudtrail.lookup_events(**lookup_kwargs)
            events = response.get("Events", [])

            for event in events:
                event_count += 1
                source = event.get("EventSource", "")
                event_name = event.get("EventName", "")
                service_prefix = source.replace(".amazonaws.com", "")
                action = f"{service_prefix}:{event_name}"
                service_action_map[service_prefix].add(action)

                detail = event.get("CloudTrailEvent")
                if detail:
                    try:
                        detail_json = json.loads(detail)
                        for resource in detail_json.get("resources", []):
                            arn = resource.get("ARN")
                            if arn:
                                resource_arns[action].add(arn)
                    except (json.JSONDecodeError, TypeError):
                        pass

            next_token = response.get("NextToken")
            if not next_token:
                break

        if event_count == 0:
            return json.dumps({
                "identity": identity_name,
                "identity_type": identity_type,
                "lookback_days": lookback_days,
                "event_count": 0,
                "policy": None,
                "message": f"No CloudTrail events found for '{identity_name}' in the last {lookback_days} days. "
                           "The identity may be inactive, or events may have aged out of free-tier event history.",
            }, indent=2)

        all_actions = sorted({a for actions in service_action_map.values() for a in actions})

        statements = []
        actions_with_specific_resources = {}
        actions_with_wildcard = []

        for action in all_actions:
            arns = resource_arns.get(action, set())
            if arns:
                actions_with_specific_resources[action] = sorted(arns)
            else:
                actions_with_wildcard.append(action)

        resource_groups = defaultdict(list)
        for action, arns in actions_with_specific_resources.items():
            key = tuple(sorted(arns))
            resource_groups[key].append(action)

        for arns_tuple, actions in sorted(resource_groups.items(), key=lambda x: x[1][0]):
            statements.append({
                "Effect": "Allow",
                "Action": sorted(actions),
                "Resource": list(arns_tuple),
            })

        if actions_with_wildcard:
            statements.append({
                "Effect": "Allow",
                "Action": sorted(actions_with_wildcard),
                "Resource": "*",
            })

        policy_document = {
            "Version": "2012-10-17",
            "Statement": statements,
        }

        service_summary = {
            svc: sorted(actions) for svc, actions in sorted(service_action_map.items())
        }

        return json.dumps({
            "identity": identity_name,
            "identity_type": identity_type,
            "lookback_days": lookback_days,
            "event_count": event_count,
            "unique_actions": len(all_actions),
            "unique_services": len(service_action_map),
            "services_used": service_summary,
            "generated_policy": policy_document,
            "notes": [
                "This policy is based on observed CloudTrail events and covers actions actually performed.",
                "CloudTrail free-tier event history retains management events for 90 days.",
                "Data events (S3 object-level, Lambda invoke) require a trail and may not appear here.",
                "Review the policy before attaching — some actions may have been one-time setup tasks.",
            ],
        }, indent=2)

    def _get_account_id(iam_client):
        """Extract the AWS account ID from any IAM ARN in the account."""
        try:
            return iam_client.get_user()["User"]["Arn"].split(":")[4]
        except Exception:
            try:
                users = iam_client.list_users(MaxItems=1)["Users"]
                if users:
                    return users[0]["Arn"].split(":")[4]
            except Exception:
                pass
        return "UNKNOWN"

    # ── Agent setup ──────────────────────────────────────────────────

    tools = [
        list_iam_users, list_iam_roles, list_iam_policies,
        get_user_policies, get_role_policies, get_policy_document,
        list_access_keys,
        create_access_key, deactivate_access_key, delete_access_key,
        store_secret, get_secret, list_secrets, delete_secret,
        rotate_and_store_key,
        security_audit, generate_credential_report,
        simulate_policy, simulate_custom_policy,
        get_unused_permissions, get_last_accessed_details,
        least_privilege_advisor,
    ]

    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    SYSTEM_PROMPT = """You are an advanced AWS IAM security agent powered by GPT-4.

You can **read** IAM state, **audit** security posture, **simulate** permissions,
**analyze** least privilege, **rotate** keys, and **manage** secrets.
Always use tools to fetch live data before answering.

CAPABILITIES:
- List and inspect IAM users, roles, policies, and access keys
- Detect stale access keys (>90 days old) and recommend rotation
- Rotate IAM access keys: create new key → store in Secrets Manager → deactivate old key
- Store, retrieve, list, and delete secrets in AWS Secrets Manager
- Run full security audits: stale keys, missing MFA, overly permissive policies
- Generate and parse the AWS credential report for account-wide health snapshots
- Simulate permissions: test "can user X do action Y on resource Z?" without trying it
- Analyze least privilege: find unused permissions via Access Advisor
- Generate least-privilege policies from CloudTrail event history for any user or role

AUDIT RULES:
- When asked to audit, run `security_audit` for a quick scan with risk scoring.
- Use `generate_credential_report` for the full AWS-native report (password age, MFA,
  key age, last login for every user). This takes ~10-30s as AWS generates it async.
- Present findings grouped by severity: CRITICAL > HIGH > MEDIUM > LOW.

POLICY SIMULATOR RULES:
- Use `simulate_policy` to answer "can user X do action Y on resource Z?" questions.
  The user provides a principal ARN, comma-separated actions, and optional resource ARNs.
- Use `simulate_custom_policy` to test a policy JSON document before attaching it.
- Always explain the verdict clearly: which actions are allowed/denied and by which policy.

LEAST PRIVILEGE RULES:
- Use `get_unused_permissions` to find services a user/role has access to but has never used.
  Recommend removing those permissions.
- Use `get_last_accessed_details` for a quick view of when each service was last accessed.
- These tools call Access Advisor which takes ~10-30s to generate results.
- When asked "what should the least privilege policy look like for <user/role>?", use
  `least_privilege_advisor` to query CloudTrail event history and generate a policy based on
  actual API usage. This builds a ready-to-use IAM policy JSON from real events.
- `least_privilege_advisor` scans up to 90 days of CloudTrail management events. Pass
  identity_type="role" for roles and identity_type="user" for users.
- Always present the generated policy as formatted JSON and remind the user to review it
  before attaching — some actions may have been one-time setup tasks that shouldn't be in
  the long-term policy.
- If no events are found, suggest the identity may be inactive or that data events (S3
  object-level, Lambda invocations) require a CloudTrail trail to be recorded.

KEY ROTATION RULES:
- AWS allows a maximum of 2 access keys per IAM user.
- `rotate_and_store_key` handles this automatically: if the user has 2 keys it deletes
  the oldest one first, then creates a new key, stores it, and deactivates the remaining old key.
- Users with NO access keys should be SKIPPED during rotation — there is nothing to rotate.
  The tool handles this and returns status "skipped", but prefer checking with `list_access_keys`
  first and only rotating users that actually have keys.
- When rotating keys for multiple users, process them one at a time.

SAFETY RULES:
- When rotating keys, always store the new credentials in Secrets Manager BEFORE
  deactivating old keys so credentials are never lost.
- Never display SecretAccessKey values in full — show only the first 4 and last 4 characters.
- If a tool returns an error, explain the issue clearly to the user and suggest next steps.

EXECUTION RULES:
- When the user confirms an action (e.g. "yes", "confirm", "go ahead"), execute it immediately
  using the data from the conversation history. Do NOT re-ask for information you already have.
- When asked to scan all users, iterate through every user, check their keys, and perform
  the requested action on each one. Summarize results at the end.
- Always use real access key IDs from tool results. NEVER fabricate or use placeholder key IDs.
- When deleting a key, always call `list_access_keys` first to get the real key ID, then
  `deactivate_access_key`, then `delete_access_key`.

CRITICAL — EXCLUSIONS AND CONSTRAINTS:
- If the user says to EXCLUDE, SKIP, or NOT touch specific users, you MUST respect that.
  Before performing any action on a user, check if they are in the exclusion list.
- Build an explicit skip list from the user's message BEFORE you start processing.
  For example: "rotate all keys but don't touch sharath" → skip_list = ["sharath"].
- After completing a bulk operation, list which users were processed AND which were skipped
  (with the reason), so the user can verify the exclusions were respected.
- When in doubt about whether a user should be included, ASK — do not assume.

Be concise, security-conscious, and proactive about flagging stale keys or risky configs."""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    print("🔐 IAM Agent (GPT-4) ready. Ask anything about your AWS IAM setup.")
    print("   Now with key rotation & Secrets Manager support.")
    print("   Type 'exit' or 'quit' to stop.\n")

    conversation_history = []

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.strip().lower() in ("exit", "quit"):
            break

        conversation_history.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": conversation_history})

        conversation_history = result["messages"]
        last_msg = conversation_history[-1]
        print(f"\nAgent: {last_msg.content}\n")


if __name__ == "__main__":
    main()
