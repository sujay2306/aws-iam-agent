import os
import json
import boto3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def main():
    load_dotenv()

    iam = boto3.client(
        "iam",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )

    @tool
    def list_iam_users() -> str:
        """Lists all IAM users in the AWS account."""
        users = iam.list_users()["Users"]
        return json.dumps([{"UserName": u["UserName"], "UserId": u["UserId"], "Arn": u["Arn"]} for u in users], indent=2)

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
        """Lists access keys for a given IAM user including their status and creation date."""
        keys = iam.list_access_keys(UserName=username)["AccessKeyMetadata"]
        return json.dumps([{
            "AccessKeyId": k["AccessKeyId"],
            "Status": k["Status"],
            "CreateDate": str(k["CreateDate"])
        } for k in keys], indent=2)

    tools = [
        list_iam_users, list_iam_roles, list_iam_policies,
        get_user_policies, get_role_policies, get_policy_document, list_access_keys,
    ]

    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AWS IAM expert assistant.
         You can list users, roles, policies, access keys, and explain permissions.
         Always use the tools to fetch live data before answering.
         Be concise and clear in your responses."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("IAM Agent ready. Ask anything about your AWS IAM setup.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = agent_executor.invoke({"input": user_input})
        print(f"\nAgent: {result['output']}")


if __name__ == "__main__":
    main()