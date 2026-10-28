"""Create (or update in place) the IAM role required by Amazon Bedrock AgentCore Runtime.

This script creates an IAM role with:
- Trust policy allowing bedrock-agentcore.amazonaws.com to assume the role
- Permissions for: Bedrock model invocation (and ApplyGuardrail), ECR image pull,
  CloudWatch Logs, DynamoDB access (for agents that need it), MCP Lambda invoke,
  AgentCore Memory data plane, AgentCore gateway invocation, and invoking the
  specialist agent runtimes (orchestrator -> specialists)

Idempotent: if the role already exists, its trust policy and inline policy are
overwritten with the current definitions (put_role_policy), so rerunning this
script is how permission changes reach an existing deployment.

Usage:
    python scripts/create_agentcore_role.py [--region us-east-1] [--role-name RetailPricingAgentCoreRole]
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROLE_NAME_DEFAULT = "RetailPricingAgentCoreRole"

TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock-agentcore.amazonaws.com"
            },
            "Action": "sts:AssumeRole",
        }
    ],
}

INLINE_POLICY_NAME = "RetailPricingAgentCorePolicy"


def get_inline_policy(account_id: str, region: str) -> dict:
    """Build the inline policy document for AgentCore Runtime agents."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockModelInvocation",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                "Resource": [
                    # Foundation models in all regions (cross-region inference
                    # profiles route to the underlying model in several regions).
                    "arn:aws:bedrock:*::foundation-model/*",
                    # Cross-region inference profiles (the us.anthropic.* IDs the
                    # agents actually invoke) live in the caller's account.
                    f"arn:aws:bedrock:{region}:{account_id}:inference-profile/us.anthropic.*",
                ],
            },
            {
                "Sid": "ECRAuthToken",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                ],
                "Resource": "*",
            },
            {
                "Sid": "ECRImagePull",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:BatchCheckLayerAvailability",
                ],
                "Resource": [
                    f"arn:aws:ecr:{region}:{account_id}:repository/retail-pricing/*",
                ],
            },
            {
                "Sid": "CloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams",
                ],
                "Resource": [
                    f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/*",
                    f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/*:*",
                ],
            },
            {
                "Sid": "DynamoDBAccess",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                    "dynamodb:BatchGetItem",
                    "dynamodb:BatchWriteItem",
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{region}:{account_id}:table/RetailPricing*",
                ],
            },
            {
                "Sid": "S3ReadAccess",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    f"arn:aws:s3:::retail-pricing-{account_id}-{region}",
                    f"arn:aws:s3:::retail-pricing-{account_id}-{region}/*",
                ],
            },
            {
                "Sid": "LambdaInvokeMcpServers",
                "Effect": "Allow",
                "Action": [
                    "lambda:InvokeFunction",
                ],
                "Resource": [
                    f"arn:aws:lambda:{region}:{account_id}:function:rdp-mcp-*",
                ],
            },
            {
                # Data-plane calls the Strands AgentCoreMemorySessionManager makes
                # (CreateEvent/GetEvent/ListEvents/DeleteEvent for session state,
                # RetrieveMemoryRecords for long-term retrieval). Scoped to the
                # memory created by scripts/setup_memory.py (ID = name + suffix).
                "Sid": "AgentCoreMemoryDataPlane",
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:CreateEvent",
                    "bedrock-agentcore:GetEvent",
                    "bedrock-agentcore:ListEvents",
                    "bedrock-agentcore:DeleteEvent",
                    "bedrock-agentcore:RetrieveMemoryRecords",
                ],
                "Resource": [
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:memory/RetailDynamicPricingMemory-*",
                ],
            },
            {
                # The orchestrator invokes the five specialist runtimes with
                # bedrock-agentcore InvokeAgentRuntime under this role. The trailing
                # wildcard also covers the .../runtime-endpoint/DEFAULT ARN.
                "Sid": "InvokeSpecialistRuntimes",
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:InvokeAgentRuntime",
                ],
                "Resource": [
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/retailPricing_*",
                ],
            },
            {
                # Gateway invocation, scoped to the gateway from scripts/setup_gateway.py.
                # The sample gateway uses authorizerType NONE, so the agents need no
                # credentials to call it; this keeps them working if the gateway is
                # switched to AWS_IAM inbound authorization.
                "Sid": "InvokeRetailPricingGateway",
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:InvokeGateway",
                ],
                "Resource": [
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:gateway/retail-pricing-gateway-*",
                ],
            },
            {
                # Needed only when a Bedrock guardrail is attached to the model
                # (see backend/agents/agentcore/guardrail_config.py).
                "Sid": "BedrockApplyGuardrail",
                "Effect": "Allow",
                "Action": [
                    "bedrock:ApplyGuardrail",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region}:{account_id}:guardrail/*",
                ],
            },
        ],
    }


def create_role(
    session: boto3.Session, role_name: str, region: str
) -> str:
    """Create the IAM role for AgentCore Runtime. Returns the role ARN."""
    iam = session.client("iam")
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]

    # Create the role
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description=(
                "IAM role for Retail Dynamic Pricing agents deployed to "
                "Amazon Bedrock AgentCore Runtime"
            ),
            Tags=[
                {"Key": "Project", "Value": "RetailDynamicPricing"},
                {"Key": "ManagedBy", "Value": "scripts/create_agentcore_role.py"},
            ],
        )
        role_arn = response["Role"]["Arn"]
        logger.info("Created IAM role: %s", role_arn)
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            response = iam.get_role(RoleName=role_name)
            role_arn = response["Role"]["Arn"]
            logger.info("Role already exists: %s", role_arn)

            # Update the trust policy in case it changed
            iam.update_assume_role_policy(
                RoleName=role_name,
                PolicyDocument=json.dumps(TRUST_POLICY),
            )
            logger.info("Updated trust policy for existing role")
        else:
            raise

    # Apply the inline policy. This runs for BOTH a new and an already-existing
    # role: put_role_policy replaces the policy document, which is how new
    # permissions reach a role created by an earlier version of this script.
    inline_policy = get_inline_policy(account_id, region)
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=INLINE_POLICY_NAME,
        PolicyDocument=json.dumps(inline_policy),
    )
    logger.info(
        "Applied inline policy %s (%d statements)",
        INLINE_POLICY_NAME, len(inline_policy["Statement"]),
    )

    # Wait for role propagation
    logger.info("Waiting for IAM role propagation (10s)...")
    time.sleep(10)

    return role_arn


def main() -> None:
    """Create the AgentCore IAM role."""
    parser = argparse.ArgumentParser(
        description="Create IAM role for Bedrock AgentCore Runtime"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--role-name",
        default=ROLE_NAME_DEFAULT,
        help=f"IAM role name (default: {ROLE_NAME_DEFAULT})",
    )
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    role_arn = create_role(session, args.role_name, args.region)

    logger.info("=" * 60)
    logger.info("IAM role ready (created, or updated in place)!")
    logger.info("Role ARN: %s", role_arn)
    logger.info("")
    logger.info("Use this ARN with deploy_agentcore.py:")
    logger.info(
        "  python scripts/deploy_agentcore.py --role-arn %s", role_arn
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
