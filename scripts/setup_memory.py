"""Create AgentCore Memory resource for the Retail Dynamic Pricing system.

This script provisions a memory resource with semantic and summary strategies
that agents use to persist pricing outcomes and learning across cycles.

Usage:
    python scripts/setup_memory.py [--region us-east-1]

Idempotent: if the memory already exists it is reused, never duplicated.

deploy.sh runs this BEFORE deploying the agents, and scripts/deploy_agentcore.py
discovers the memory by name and sets AGENTCORE_MEMORY_ID on every agent runtime
automatically. The printed AGENTCORE_MEMORY_ID export is only needed for running
an agent container by hand.
"""

from __future__ import annotations

import argparse
import sys
import time

from bedrock_agentcore.memory import MemoryClient

# Memory name. AgentCore appends a random suffix to form the memory ID
# (e.g. RetailDynamicPricingMemory-Ab12Cd34Ef).
MEMORY_NAME = "RetailDynamicPricingMemory"

_WAIT_TIMEOUT_SECONDS = 300
_WAIT_INTERVAL_SECONDS = 10


def _memory_id(summary: dict) -> str:
    return summary.get("id") or summary.get("memoryId") or ""


def _wait_until_active(client: MemoryClient, memory_id: str) -> None:
    """Poll a memory that is still being created/updated until it is ACTIVE."""
    deadline = time.monotonic() + _WAIT_TIMEOUT_SECONDS
    status = "CREATING"
    while time.monotonic() < deadline:
        status = client.gmcp_client.get_memory(memoryId=memory_id)["memory"].get("status", "")
        if status == "ACTIVE":
            return
        if status not in ("CREATING", "UPDATING"):
            break
        print(f"  Memory status: {status}, waiting...")
        time.sleep(_WAIT_INTERVAL_SECONDS)
    raise RuntimeError(
        f"Memory {memory_id} is {status}, not ACTIVE. Refusing to create a duplicate; "
        "wait for it to finish or investigate it in the AgentCore console."
    )


def find_existing_memory(client: MemoryClient, name: str = MEMORY_NAME) -> str | None:
    """Return the ID of an existing memory created for ``name``, or None.

    list_memories() returns a list of summaries (id, arn, status, timestamps)
    with no ``name`` field, and the memory ID is the name plus a random suffix,
    so the match is on the ID prefix. Errors from the listing propagate: falling
    through to create after a failed lookup would risk a duplicate memory.
    """
    matches = [
        m for m in client.list_memories(max_results=1000)
        if _memory_id(m).startswith(f"{name}-")
    ]
    if not matches:
        return None

    # Prefer a finished memory if there are several (e.g. left over from earlier runs).
    matches.sort(key=lambda m: m.get("status") != "ACTIVE")
    chosen = matches[0]
    memory_id = _memory_id(chosen)
    if len(matches) > 1:
        print(
            f"  Warning: {len(matches)} memories match '{name}'; reusing {memory_id}. "
            "Review the others in the AgentCore console."
        )
    if chosen.get("status") != "ACTIVE":
        _wait_until_active(client, memory_id)
    return memory_id


def setup_memory(region: str = "us-east-1") -> str:
    """Create the AgentCore Memory resource.

    Idempotent — if the memory already exists, returns the existing memory ID
    instead of creating a second one.

    Args:
        region: AWS region to create the memory resource in.

    Returns:
        The memory resource ID.
    """
    client = MemoryClient(region_name=region)

    print(f"Creating AgentCore Memory in {region}...")

    # Check if memory already exists
    memory_id = find_existing_memory(client)
    if memory_id:
        print(f"  Memory already exists: {memory_id}")
        print(f"\nExport this for your agent containers:")
        print(f"  export AGENTCORE_MEMORY_ID={memory_id}")
        return memory_id

    memory = client.create_memory_and_wait(
        name=MEMORY_NAME,
        description=(
            "Memory for retail dynamic pricing agents - stores pricing outcomes, "
            "competitive intelligence, demand patterns, and learning across cycles."
        ),
        strategies=[
            {
                "semanticMemoryStrategy": {
                    "name": "PricingOutcomes",
                    "namespaceTemplates": ["/pricing/{actorId}/{sessionId}/"],
                }
            },
            {
                "summaryMemoryStrategy": {
                    "name": "CycleSummarizer",
                    "namespaceTemplates": ["/summaries/{actorId}/{sessionId}/"],
                }
            },
        ],
    )

    memory_id = memory.get("id")
    print(f"\nMemory created successfully!")
    print(f"  Memory ID: {memory_id}")
    print(f"  Region:    {region}")
    print(f"\nExport this for your agent containers:")
    print(f"  export AGENTCORE_MEMORY_ID={memory_id}")

    return memory_id


def main():
    parser = argparse.ArgumentParser(
        description="Create AgentCore Memory for Retail Dynamic Pricing"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    args = parser.parse_args()

    try:
        setup_memory(region=args.region)
    except Exception as e:
        print(f"Error creating memory: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
