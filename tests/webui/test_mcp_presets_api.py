from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nanobot.agent.agent_plugins import AGENT_PLUGIN_MCP_SCHEMA, AGENT_PLUGIN_SCHEMA
from nanobot.config.loader import load_config
from nanobot.webui.mcp_presets_api import (
    McpPresetError,
    custom_mcp_action,
    mcp_presets_action,
    mcp_presets_payload,
    mcp_presets_settings_action,
    mcp_presets_test_action,
    normalize_mcp_preset_mentions,
)


def _use_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"agents": {"defaults": {"workspace": str(tmp_path / "workspace")}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)


def _write_agent_plugin(workspace: Path) -> None:
    root = workspace / "plugins" / "desktop"
    command = root / "bin" / "server"
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "plugin.json").write_text(
        json.dumps(
            {
                "$schema": AGENT_PLUGIN_SCHEMA,
                "name": "desktop",
                "description": "Control the local desktop.",
                "extensions": {
                    "dev.nanobot": {
                        "displayName": "Desktop Control",
                        "accentColor": "#ff7a1a",
                        "permissions": ["screen-recording"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": AGENT_PLUGIN_MCP_SCHEMA,
                "mcpServers": {
                    "desktop": {"type": "stdio", "command": "./bin/server"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_mcp_presets_payload_lists_supported_cards(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(tmp_path, monkeypatch)

    payload = mcp_presets_payload()
    names = {preset["name"] for preset in payload["presets"]}

    assert {
        "browserbase",
        "playwright",
        "github",
        "figma",
        "context7",
        "firecrawl",
        "parallel-search",
        "exa",
        "microsoft-learn",
        "aws-docs",
        "brave-search",
        "postman",
    }.issubset(names)
    browserbase = next(preset for preset in payload["presets"] if preset["name"] == "browserbase")
    assert browserbase["installed"] is False
    assert browserbase["install_supported"] is True
    assert browserbase["required_fields"][0]["configured"] is False
    assert "browserbaseApiKey" not in browserbase["connection_summary"]
    manifest = browserbase["manifest"]
    assert manifest["schema"] == "agent-app.v1"
    assert manifest["id"] == "browserbase"
    assert manifest["source"] == "mcp-preset"
    assert manifest["capabilities"][0]["type"] == "mcp"
    assert manifest["capabilities"][0]["transport"] == "streamableHttp"
    assert manifest["install"]["strategy"] == "config"
    assert manifest["remove"]["verification"] == ["config_absent"]
    assert manifest["trust"]["review_status"] == "builtin_preset"


def test_agent_plugin_reuses_mcp_catalog_and_runtime_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)
    _write_agent_plugin(load_config().workspace_path)

    row = next(item for item in mcp_presets_payload()["presets"] if item["source"] == "agent-plugin")
    assert row["name"] == "plugin-desktop"
    assert row["display_name"] == "Desktop Control"
    assert row["installed"] is True
    assert row["configured"] is False

    async def reload() -> dict[str, object]:
        return {"ok": True, "message": "MCP reloaded.", "requires_restart": False}

    enabled = asyncio.run(
        mcp_presets_settings_action(
            "enable",
            {"name": ["plugin-desktop"]},
            reload_mcp=reload,
        )
    )
    enabled_row = next(item for item in enabled["presets"] if item["name"] == "plugin-desktop")
    assert enabled_row["configured"] is True
    assert enabled["requires_restart"] is False

    disabled = asyncio.run(
        mcp_presets_settings_action(
            "remove",
            {"name": ["plugin-desktop"]},
            reload_mcp=reload,
        )
    )
    disabled_row = next(item for item in disabled["presets"] if item["name"] == "plugin-desktop")
    assert disabled_row["configured"] is False


def test_explicit_mcp_config_wins_over_plugin_catalog_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tools"] = {
        "mcpServers": {"plugin-desktop": {"type": "stdio", "command": "echo"}}
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _write_agent_plugin(load_config().workspace_path)

    rows = [
        item for item in mcp_presets_payload()["presets"] if item["name"] == "plugin-desktop"
    ]

    assert len(rows) == 1
    assert rows[0]["source"] == "custom"


def test_enable_browserbase_writes_scrubbed_config_payload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)

    payload = mcp_presets_action(
        "enable",
        {
            "name": ["browserbase"],
            "browserbase_api_key": ["bb_live_secret"],
        },
    )

    assert payload["requires_restart"] is True
    assert payload["last_action"]["ok"] is True
    assert payload["last_action"]["installed"] is True
    assert payload["last_action"]["verification"] == ["config_present"]
    preset = next(row for row in payload["presets"] if row["name"] == "browserbase")
    assert preset["installed"] is True
    assert preset["configured"] is True
    assert "bb_live_secret" not in str(payload)
    config = load_config()
    assert "browserbaseApiKey=bb_live_secret" in config.tools.mcp_servers["browserbase"].url


def test_enable_requires_missing_secret(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(tmp_path, monkeypatch)

    with pytest.raises(McpPresetError) as exc:
        mcp_presets_action("enable", {"name": ["browserbase"]})

    assert exc.value.status == 400
    assert "Browserbase API key" in exc.value.message


def test_enable_context7_optional_api_key_appends_arg(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)

    payload = mcp_presets_action(
        "enable",
        {
            "name": ["context7"],
            "context7_api_key": ["ctx7_secret"],
        },
    )

    assert "ctx7_secret" not in str(payload)
    row = next(item for item in payload["presets"] if item["name"] == "context7")
    assert row["configured"] is True
    config = load_config()
    assert config.tools.mcp_servers["context7"].args == [
        "-y",
        "@upstash/context7-mcp@latest",
        "--api-key",
        "ctx7_secret",
    ]


def test_enable_stdio_preset_uses_config_scoped_cwd(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)

    mcp_presets_action("enable", {"name": ["playwright"]})

    config = load_config()
    cwd = config.tools.mcp_servers["playwright"].cwd
    assert cwd == str(tmp_path / "mcp" / "playwright")
    assert (tmp_path / "mcp" / "playwright").is_dir()


def test_enable_no_auth_remote_presets_write_url(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(tmp_path, monkeypatch)

    mcp_presets_action("enable", {"name": ["microsoft-learn"]})
    mcp_presets_action("enable", {"name": ["exa"]})
    mcp_presets_action("enable", {"name": ["firecrawl"]})
    mcp_presets_action("enable", {"name": ["parallel-search"]})

    config = load_config()
    assert config.tools.mcp_servers["microsoft-learn"].url == "https://learn.microsoft.com/api/mcp"
    assert config.tools.mcp_servers["exa"].url == "https://mcp.exa.ai/mcp"
    assert config.tools.mcp_servers["firecrawl"].url == "https://mcp.firecrawl.dev/v2/mcp"
    assert config.tools.mcp_servers["parallel-search"].url == "https://search.parallel.ai/mcp"


def test_firecrawl_preset_is_keyless(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(tmp_path, monkeypatch)

    payload = mcp_presets_action("enable", {"name": ["firecrawl"]})

    row = next(item for item in payload["presets"] if item["name"] == "firecrawl")
    assert row["transport"] == "streamableHttp"
    assert row["requires"] == "Network access"
    assert row["required_fields"] == []
    assert row["configured"] is True
    assert "Keyless" in row["note"]
    config = load_config()
    assert config.tools.mcp_servers["firecrawl"].type == "streamableHttp"
    assert config.tools.mcp_servers["firecrawl"].url == "https://mcp.firecrawl.dev/v2/mcp"
    assert config.tools.mcp_servers["firecrawl"].env == {}


def test_parallel_search_preset_is_keyless_and_tool_limited(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)

    payload = mcp_presets_action("enable", {"name": ["parallel-search"]})

    row = next(item for item in payload["presets"] if item["name"] == "parallel-search")
    assert row["transport"] == "streamableHttp"
    assert row["requires"] == "Network access"
    assert row["required_fields"] == []
    assert row["configured"] is True
    assert "no API key" in row["note"]
    config = load_config()
    server = config.tools.mcp_servers["parallel-search"]
    assert server.type == "streamableHttp"
    assert server.url == "https://search.parallel.ai/mcp"
    assert server.enabled_tools == ["web_search", "web_fetch"]
    assert server.headers == {}


def test_remove_mcp_preset_updates_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(tmp_path, monkeypatch)
    mcp_presets_action("enable", {"name": ["playwright"]})
    managed_cwd = tmp_path / "mcp" / "playwright"
    (managed_cwd / "cache.txt").write_text("managed runtime data", encoding="utf-8")

    payload = mcp_presets_action("remove", {"name": ["playwright"]})

    assert payload["requires_restart"] is True
    assert payload["last_action"]["ok"] is True
    assert payload["last_action"]["removed"] is True
    assert payload["last_action"]["managed_paths_removed"] == ["runtime:mcp/playwright"]
    assert not managed_cwd.exists()
    config = load_config()
    assert "playwright" not in config.tools.mcp_servers


def test_remove_custom_mcp_server_preserves_user_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(tmp_path, monkeypatch)
    user_cwd = tmp_path / "user-cwd"
    user_cwd.mkdir()
    custom_mcp_action(
        "custom",
        {
            "name": ["internal-docs"],
            "transport": ["stdio"],
            "command": ["node"],
            "args": ['["server.js"]'],
            "cwd": [str(user_cwd)],
        },
    )

    payload = mcp_presets_action("remove", {"name": ["internal-docs"]})

    assert payload["last_action"]["ok"] is True
    assert user_cwd.exists()
    config = load_config()
    assert "internal-docs" not in config.tools.mcp_servers


def test_test_mcp_preset_reports_missing_dependency(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)
    mcp_presets_action("enable", {"name": ["playwright"]})
    monkeypatch.setattr("nanobot.webui.mcp_presets_api.shutil.which", lambda _command: None)

    payload = asyncio.run(mcp_presets_test_action({"name": ["playwright"]}))

    assert payload["last_action"]["ok"] is False
    assert "npx" in payload["last_action"]["message"]


def test_test_mcp_preset_connects_and_reports_tools(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)
    mcp_presets_action("enable", {"name": ["playwright"]})

    class FakeStack:
        async def aclose(self) -> None:
            return None

    async def fake_connect(servers, registry):
        assert list(servers) == ["playwright"]

        class FakeTool:
            name = "mcp_playwright_browser_navigate"

            def to_schema(self):
                return {"name": self.name, "description": "", "parameters": {}}

        registry.register(FakeTool())
        return {"playwright": FakeStack()}

    monkeypatch.setattr("nanobot.agent.tools.mcp.connect_mcp_servers", fake_connect)

    payload = asyncio.run(mcp_presets_test_action({"name": ["playwright"]}))

    assert payload["last_action"]["ok"] is True
    assert payload["last_action"]["tool_count"] == 1
    assert payload["last_action"]["tool_names"] == ["mcp_playwright_browser_navigate"]


def test_test_mcp_preset_scrubs_connection_errors(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)
    mcp_presets_action(
        "enable",
        {
            "name": ["browserbase"],
            "browserbase_api_key": ["bb_live_secret"],
        },
    )

    async def fake_connect(_servers, _registry):
        raise RuntimeError("failed https://mcp.browserbase.com/mcp?browserbaseApiKey=bb_live_secret")

    monkeypatch.setattr("nanobot.agent.tools.mcp.connect_mcp_servers", fake_connect)

    payload = asyncio.run(mcp_presets_test_action({"name": ["browserbase"]}))

    assert payload["last_action"]["ok"] is False
    assert "bb_live_secret" not in str(payload)
    assert "<redacted>" in payload["last_action"]["error"]


def test_unlisted_oauth_placeholder_is_not_enabled(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(tmp_path, monkeypatch)

    with pytest.raises(McpPresetError) as exc:
        mcp_presets_action("enable", {"name": ["linear"]})

    assert exc.value.status == 404


def test_normalize_mcp_preset_mentions_keeps_known_presets_only() -> None:
    payload = normalize_mcp_preset_mentions([
        {
            "name": "browserbase",
            "display_name": "Browserbase",
            "transport": "streamableHttp",
            "configured": True,
            "logo_url": "https://example.invalid/logo.svg",
        },
        {"name": "totally-unknown"},
        "bad",
    ])

    assert payload == [{
        "name": "browserbase",
        "display_name": "Browserbase",
        "transport": "streamableHttp",
        "configured": True,
        "logo_url": "https://example.invalid/logo.svg",
    }]


def test_custom_mcp_server_writes_config_and_catalog_row(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)

    payload = custom_mcp_action(
        "custom",
        {
            "name": ["internal-docs"],
            "transport": ["stdio"],
            "command": ["node"],
            "args": ['["server.js"]'],
            "env": ['{"DOCS_TOKEN":"docs-secret-value"}'],
            "tool_timeout": ["45"],
        },
    )

    assert payload["requires_restart"] is True
    row = next(item for item in payload["presets"] if item["name"] == "internal-docs")
    assert row["source"] == "custom"
    assert row["transport"] == "stdio"
    assert row["connection_summary"] == "node server.js"
    assert row["manifest"]["schema"] == "agent-app.v1"
    assert row["manifest"]["source"] == "mcp-custom"
    assert row["manifest"]["capabilities"][0]["command"] == "node"
    assert "server.js" not in str(row["manifest"])
    assert "docs-secret-value" not in str(payload)
    config = load_config()
    assert config.tools.mcp_servers["internal-docs"].args == ["server.js"]
    assert config.tools.mcp_servers["internal-docs"].env["DOCS_TOKEN"] == "docs-secret-value"


def test_import_mcp_config_and_tool_allowlist(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)

    payload = custom_mcp_action(
        "import",
        {
            "config": [
                (
                    '{"mcpServers":{'
                    '"docs":{"command":"npx","args":["-y","docs-mcp"],"env":{"API_KEY":"config-secret-value"}},'
                    '"remote-docs":{"transport":"sse","url":"https://example.com/sse"}'
                    '}}'
                )
            ],
        },
    )

    assert payload["last_action"]["message"] == "Imported 2 MCP server(s)."
    config = load_config()
    assert config.tools.mcp_servers["docs"].command == "npx"
    assert config.tools.mcp_servers["docs"].args == ["-y", "docs-mcp"]
    assert config.tools.mcp_servers["remote-docs"].type == "sse"
    assert config.tools.mcp_servers["remote-docs"].url == "https://example.com/sse"
    assert config.tools.mcp_servers["docs"].env["API_KEY"] == "config-secret-value"
    assert "config-secret-value" not in str(payload)

    payload = custom_mcp_action(
        "tools",
        {
            "name": ["docs"],
            "enabled_tools": ['["mcp_docs_search"]'],
        },
    )

    row = next(item for item in payload["presets"] if item["name"] == "docs")
    assert row["enabled_tools"] == ["mcp_docs_search"]
    assert load_config().tools.mcp_servers["docs"].enabled_tools == ["mcp_docs_search"]

    payload = custom_mcp_action(
        "tools",
        {
            "name": ["docs"],
            "enabled_tools": ["[]"],
        },
    )

    row = next(item for item in payload["presets"] if item["name"] == "docs")
    assert row["enabled_tools"] == []
    assert load_config().tools.mcp_servers["docs"].enabled_tools == []


def test_normalize_mcp_preset_mentions_accepts_configured_custom_server(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(tmp_path, monkeypatch)
    custom_mcp_action(
        "custom",
        {
            "name": ["docs"],
            "transport": ["streamableHttp"],
            "url": ["https://example.com/mcp"],
        },
    )

    payload = normalize_mcp_preset_mentions([
        {"name": "docs", "display_name": "Docs", "transport": "streamableHttp"},
    ])

    assert payload == [{"name": "docs", "display_name": "Docs", "transport": "streamableHttp"}]
