import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from nanobot.agent import agent_plugins
from nanobot.agent.agent_plugins import (
    AGENT_PLUGIN_MCP_SCHEMA,
    AGENT_PLUGIN_SCHEMA,
    agent_plugin_mcp_servers,
    agent_plugins_payload,
    discover_agent_plugin_skills,
    set_agent_plugin_enabled,
)
from nanobot.agent.skills import SkillsLoader


def _write_skill(root: Path, name: str, *, description: str = "Plugin skill.") -> Path:
    skill = root / "skills" / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return skill


def _write_plugin(
    workspace: Path,
    directory: str,
    *,
    name: str | None = None,
    manifest: dict[str, object] | None = None,
) -> Path:
    root = workspace / "plugins" / directory
    root.mkdir(parents=True)
    payload = manifest or {
        "$schema": AGENT_PLUGIN_SCHEMA,
        "name": name or directory,
    }
    (root / "plugin.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_skills_loader_discovers_agent_plugin_skill(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path, "acme-tools")
    _write_skill(plugin, "release-notes", description="Draft release notes from changes.")

    loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")

    assert loader.list_skills() == [
        {
            "name": "release-notes",
            "path": str(plugin / "skills" / "release-notes" / "SKILL.md"),
            "source": "plugin",
            "plugin": "acme-tools",
        }
    ]
    assert loader.get_explicitly_invoked_skills("Use $release-notes") == ["release-notes"]
    assert "Draft release notes" in (loader.load_skill("release-notes") or "")
    assert "### Agent Plugin skills" in loader.build_skills_summary()
    assert "`acme-tools/skills/release-notes/SKILL.md`" in loader.build_skills_summary()


def test_skills_loader_sees_plugin_installed_after_startup(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
    assert loader.list_skills() == []

    plugin = _write_plugin(tmp_path, "acme-tools")
    _write_skill(plugin, "release-notes")

    assert [entry["name"] for entry in loader.list_skills()] == ["release-notes"]

    shutil.rmtree(plugin)

    assert loader.list_skills() == []
    assert loader.build_skills_summary() == ""


def test_agent_plugin_skills_are_direct_children_only(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path, "acme-tools")
    _write_skill(plugin, "direct")
    nested = plugin / "skills" / "group" / "nested"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: nested\ndescription: Nested skill.\n---\n",
        encoding="utf-8",
    )

    assert [skill.name for skill in discover_agent_plugin_skills(tmp_path)] == ["direct"]


@pytest.mark.parametrize(
    "manifest",
    [
        {"$schema": "https://agent-plugins.org/schemas/2.0.0/plugin.schema.json", "name": "demo"},
        {"$schema": AGENT_PLUGIN_SCHEMA, "name": "Bad-Name"},
        {"$schema": AGENT_PLUGIN_SCHEMA, "name": "demo", "author": None},
        {"$schema": AGENT_PLUGIN_SCHEMA, "name": "demo", "keywords": None},
    ],
)
def test_invalid_agent_plugin_manifest_is_skipped(
    tmp_path: Path,
    manifest: dict[str, object],
) -> None:
    plugin = _write_plugin(tmp_path, "demo", manifest=manifest)
    _write_skill(plugin, "example")

    assert discover_agent_plugin_skills(tmp_path) == []


def test_unknown_manifest_fields_and_non_object_extensions_are_ignored(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path,
        "demo",
        manifest={
            "$schema": AGENT_PLUGIN_SCHEMA,
            "name": "demo",
            "futureField": True,
            "extensions": "invalid but non-fatal",
        },
    )
    _write_skill(plugin, "example")

    assert [skill.name for skill in discover_agent_plugin_skills(tmp_path)] == ["example"]


@pytest.mark.parametrize(
    ("skill_name", "frontmatter"),
    [
        ("wrong-directory", "name: another\ndescription: Mismatch."),
        ("missing-description", "name: missing-description"),
        ("Bad-Name", "name: Bad-Name\ndescription: Invalid name."),
    ],
)
def test_invalid_agent_skill_is_skipped(
    tmp_path: Path,
    skill_name: str,
    frontmatter: str,
) -> None:
    plugin = _write_plugin(tmp_path, "demo")
    skill = plugin / "skills" / skill_name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n", encoding="utf-8")

    assert discover_agent_plugin_skills(tmp_path) == []


def test_workspace_skill_overrides_plugin_skill(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path, "demo")
    _write_skill(plugin, "shared", description="Plugin version.")
    workspace_skill = tmp_path / "skills" / "shared"
    workspace_skill.mkdir(parents=True)
    (workspace_skill / "SKILL.md").write_text(
        "---\nname: shared\ndescription: Workspace version.\n---\n",
        encoding="utf-8",
    )

    loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")

    assert [entry["source"] for entry in loader.list_skills()] == ["workspace"]
    assert "Workspace version" in (loader.load_skill("shared") or "")


def test_plugin_skill_symlink_cannot_escape_plugin_root(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path, "demo")
    outside = tmp_path / "outside"
    _write_skill(outside, "escaped")
    skills_root = plugin / "skills"
    skills_root.mkdir()
    try:
        (skills_root / "escaped").symlink_to(
            outside / "skills" / "escaped",
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    assert discover_agent_plugin_skills(tmp_path) == []


def test_plugin_mcp_requires_explicit_enable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_plugins,
        "get_config_path",
        lambda: tmp_path / "config" / "config.json",
    )
    plugin = _write_plugin(tmp_path, "desktop")
    executable = plugin / "bin" / "server"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    (plugin / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": AGENT_PLUGIN_MCP_SCHEMA,
                "mcpServers": {
                    "desktop": {
                        "type": "stdio",
                        "command": "./bin/server",
                        "args": ["--data", "${PLUGIN_DATA}/state"],
                        "cwd": "${PLUGIN_ROOT}",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert agent_plugin_mcp_servers(tmp_path) == {}
    set_agent_plugin_enabled(tmp_path, "desktop", True)

    servers = agent_plugin_mcp_servers(tmp_path)
    server = servers["desktop"]
    assert server.command == str(executable)
    assert server.cwd == str(plugin)
    assert server.env["PLUGIN_ROOT"] == str(plugin)
    assert server.args[0] == "--data"
    assert server.args[1].endswith("/state")

    set_agent_plugin_enabled(tmp_path, "desktop", False)
    assert agent_plugin_mcp_servers(tmp_path) == {}


def test_plugin_setup_command_runs_once_per_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_plugins,
        "get_config_path",
        lambda: tmp_path / "config" / "config.json",
    )
    monkeypatch.setenv("NANOBOT_TEST_SECRET", "do-not-inherit")
    plugin = _write_plugin(
        tmp_path,
        "desktop",
        manifest={
            "$schema": AGENT_PLUGIN_SCHEMA,
            "name": "desktop",
            "version": "1.2.3",
            "extensions": {"dev.nanobot": {"installCommand": ["./bin/install"]}},
        },
    )
    executable = plugin / "bin" / "install"
    executable.parent.mkdir()
    executable.write_text("setup", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def run(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, cast(dict[str, str], kwargs["env"])))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(agent_plugins.subprocess, "run", run)

    set_agent_plugin_enabled(tmp_path, "desktop", True)
    set_agent_plugin_enabled(tmp_path, "desktop", False)
    set_agent_plugin_enabled(tmp_path, "desktop", True)

    assert len(calls) == 1
    assert calls[0][0] == (str(executable),)
    assert calls[0][1]["PLUGIN_ROOT"] == str(plugin)
    assert "NANOBOT_TEST_SECRET" not in calls[0][1]
    assert agent_plugins_payload(tmp_path)["plugins"][0]["setup_required"] is False


def test_invalid_plugin_mcp_entries_do_not_block_valid_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_plugins,
        "get_config_path",
        lambda: tmp_path / "config" / "config.json",
    )
    plugin = _write_plugin(tmp_path, "network")
    executable = plugin / "bin" / "server"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    (plugin / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": AGENT_PLUGIN_MCP_SCHEMA,
                "mcpServers": {
                    "public-http": {"type": "streamable-http", "url": "http://example.com/mcp"},
                    "local": {"type": "stdio", "command": "./bin/server"},
                    "escape": {"type": "stdio", "command": "../outside"},
                },
            }
        ),
        encoding="utf-8",
    )
    set_agent_plugin_enabled(tmp_path, "network", True)

    assert list(agent_plugin_mcp_servers(tmp_path)) == ["network"]


def test_plugin_state_symlink_cannot_escape_config_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config"
    outside = tmp_path / "outside"
    config.mkdir()
    outside.mkdir()
    try:
        (config / "plugin-data").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    monkeypatch.setattr(
        agent_plugins,
        "get_config_path",
        lambda: config / "config.json",
    )
    _write_plugin(tmp_path, "desktop")

    with pytest.raises(RuntimeError, match="escapes the nanobot config directory"):
        set_agent_plugin_enabled(tmp_path, "desktop", True)
