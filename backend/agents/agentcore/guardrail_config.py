"""Bedrock Guardrails configuration for Retail Dynamic Pricing agents.

All agents in the pricing system use this guardrail to prevent:
- Predatory pricing strategies
- Price fixing and collusion
- Discriminatory pricing based on protected characteristics
- Price gouging during emergencies

Default guardrail ID: mgu29odtpk6m
Default version: 4

The demo does not create a guardrail of its own, so the defaults above only
work if that guardrail exists in the target account. scripts/deploy_agentcore.py
checks for it at deploy time and sets one of the following on the runtimes:

    BEDROCK_GUARDRAIL_ID / BEDROCK_GUARDRAIL_VERSION
        Guardrail to attach (override the defaults above).
    DISABLE_BEDROCK_GUARDRAILS=true
        ROLLBACK / kill switch: get_guardrail_model_kwargs() returns an empty
        dict, effectively bypassing the guardrail without code changes.

The guardrail is attached to the Bedrock *model* (BedrockModel guardrail_id /
guardrail_version / guardrail_trace). strands.Agent itself has no guardrail
parameter, so the kwargs must not be passed to Agent(...).
"""

import os

# Default guardrail identifiers (used when BEDROCK_GUARDRAIL_ID / _VERSION are unset)
GUARDRAIL_ID = "mgu29odtpk6m"
GUARDRAIL_VERSION = "4"


def get_guardrail_model_kwargs() -> dict:
    """Return BedrockModel keyword arguments that attach the guardrail.

    Returns an empty dict if DISABLE_BEDROCK_GUARDRAILS=true is set or the
    guardrail ID is empty, allowing quick rollback without redeployment.

    Usage with Strands:
        from strands import Agent
        from strands.models import BedrockModel
        from backend.agents.agentcore.guardrail_config import get_guardrail_model_kwargs
        from shared.model_config import SPECIALIST_MODEL

        model = BedrockModel(model_id=SPECIALIST_MODEL, **get_guardrail_model_kwargs())
        agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=tools)
    """
    if os.environ.get("DISABLE_BEDROCK_GUARDRAILS", "").lower() == "true":
        return {}

    guardrail_id = get_guardrail_id()
    if not guardrail_id:
        return {}

    return {
        "guardrail_id": guardrail_id,
        "guardrail_version": get_guardrail_version(),
        "guardrail_trace": "enabled",
    }


def get_guardrail_id() -> str:
    """Return the guardrail ID for direct API usage."""
    return os.environ.get("BEDROCK_GUARDRAIL_ID", GUARDRAIL_ID)


def get_guardrail_version() -> str:
    """Return the guardrail version for direct API usage."""
    return os.environ.get("BEDROCK_GUARDRAIL_VERSION", GUARDRAIL_VERSION)
