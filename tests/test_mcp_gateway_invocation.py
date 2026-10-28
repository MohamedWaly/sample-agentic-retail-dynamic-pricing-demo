"""Tests for the AgentCore Gateway invocation shape of the MCP server Lambdas.

AgentCore Gateway invokes a Lambda target with the tool *arguments* as the event
and the tool name in ``context.client_context.custom["bedrockAgentCoreToolName"]``
(formatted ``<target>___<tool>``). The handlers must accept that shape in addition
to the direct-invoke shape ``{"tool": ..., "params": {...}}``, which the existing
handler tests cover.
"""
import inspect
import json
from types import SimpleNamespace

import pytest

from backend.mcp_servers.competitor_api import handler as competitor_api
from backend.mcp_servers.cost_finance import handler as cost_finance
from backend.mcp_servers.erp_pos import handler as erp_pos
from backend.mcp_servers.market_signals import handler as market_signals

HANDLER_MODULES = [competitor_api, erp_pos, market_signals, cost_finance]


def gateway_context(tool_name):
    """Build a stand-in for the Lambda context an AgentCore Gateway passes."""
    return SimpleNamespace(
        client_context=SimpleNamespace(
            custom={
                "bedrockAgentCoreMessageVersion": "1.0",
                "bedrockAgentCoreToolName": tool_name,
            }
        )
    )


# (module, gateway tool name, tool arguments, key expected in response data)
GATEWAY_CASES = [
    (competitor_api, "competitor-api___get_competitor_prices", {"product_id": "ELEC-001"}, "competitors"),
    (competitor_api, "competitor-api___get_price_history", {"product_id": "ELEC-001", "days": 7}, "history"),
    (erp_pos, "erp-pos___get_inventory_levels", {"product_id": "PROD-001"}, "locations"),
    (erp_pos, "erp-pos___get_sales_history", {"product_id": "PROD-001", "period": "weekly"}, "salesHistory"),
    (market_signals, "market-signals___get_market_trends", {"category": "electronics"}, "growthRate"),
    (market_signals, "market-signals___get_macro_indicators", {"region": "US"}, "unemploymentRate"),
    (cost_finance, "cost-finance___get_cost_structure", {"category": "electronics"}, "cost_breakdown"),
    (cost_finance, "cost-finance___get_financial_constraints", {"channel": "online"}, "channel_rules"),
]


class TestGatewayInvocationShape:
    @pytest.mark.parametrize("module,tool_name,args,data_key", GATEWAY_CASES)
    def test_gateway_call_returns_success(self, module, tool_name, args, data_key):
        result = module.handler(dict(args), gateway_context(tool_name))

        body = json.loads(result["body"])
        assert body["status"] == "success", body.get("error")
        assert data_key in body["data"]
        assert body.get("error", {}).get("code") != "MISSING_TOOL"

    def test_tool_arguments_are_forwarded_to_the_tool(self):
        result = competitor_api.handler(
            {"product_id": "ELEC-003"},
            gateway_context("competitor-api___get_competitor_prices"),
        )

        assert json.loads(result["body"])["data"]["productId"] == "ELEC-003"

    @pytest.mark.parametrize("module", HANDLER_MODULES)
    def test_unknown_gateway_tool_is_rejected(self, module):
        result = module.handler({}, gateway_context("target___no_such_tool"))

        body = json.loads(result["body"])
        assert body["status"] == "error"
        assert body["error"]["code"] == "UNKNOWN_TOOL"

    def test_tool_name_without_target_prefix_is_accepted(self):
        result = cost_finance.handler(
            {"category": "electronics"}, gateway_context("get_cost_structure")
        )

        assert json.loads(result["body"])["status"] == "success"

    def test_explicit_tool_field_takes_precedence_over_context(self):
        event = {"tool": "get_competitor_prices", "params": {"product_id": "ELEC-001"}}

        result = competitor_api.handler(event, gateway_context("target___get_price_history"))

        assert "competitors" in json.loads(result["body"])["data"]


class TestNoGatewayMetadata:
    """Without gateway metadata a missing tool is still reported as MISSING_TOOL."""

    @pytest.mark.parametrize("module", HANDLER_MODULES)
    @pytest.mark.parametrize(
        "context",
        [
            None,
            SimpleNamespace(),
            SimpleNamespace(client_context=None),
            SimpleNamespace(client_context=SimpleNamespace(custom=None)),
            SimpleNamespace(client_context=SimpleNamespace(custom={})),
            SimpleNamespace(client_context=SimpleNamespace(custom={"bedrockAgentCoreToolName": ""})),
            SimpleNamespace(client_context=SimpleNamespace(custom={"bedrockAgentCoreToolName": 42})),
        ],
        ids=[
            "none",
            "no-client-context-attr",
            "client-context-none",
            "custom-none",
            "custom-empty",
            "tool-name-empty",
            "tool-name-not-str",
        ],
    )
    def test_missing_tool_reported(self, module, context):
        result = module.handler({"product_id": "ELEC-001"}, context)

        body = json.loads(result["body"])
        assert body["status"] == "error"
        assert body["error"]["code"] == "MISSING_TOOL"


def test_gateway_tool_name_helpers_stay_identical():
    """Each Lambda ships on its own, so the helper is copied per handler. Keep the copies in sync."""
    sources = {inspect.getsource(m._gateway_tool_name) for m in HANDLER_MODULES}
    assert len(sources) == 1
