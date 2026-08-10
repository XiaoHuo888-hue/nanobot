"""Load and activate locally installed Agent Plugin packages."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from filelock import FileLock
from loguru import logger
from pydantic import ValidationError

from nanobot.agent.skills import parse_skill_metadata, valid_skill_metadata
from nanobot.config.loader import get_config_path
from nanobot.config.schema import MCPServerConfig

AGENT_PLUGIN_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
AGENT_PLUGIN_MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"

_PLUGIN_NAME = re.compile(r"^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_MCP_SERVER_FIELDS = {"type", "command", "args", "env", "cwd"}
_SETUP_ENV = {"HOME", "LANG", "LC_ALL", "LOGNAME", "PATH", "SHELL", "TMPDIR", "USER"}
_SETUP_TIMEOUT_SECONDS = 600
_MAX_LOGO_BYTES = 256 * 1024


@dataclass(frozen=True)
class AgentPluginSkill:
    """One skill supplied by a valid Agent Plugins v1 package."""

    name: str
    path: Path
    plugin: str


@dataclass(frozen=True)
class AgentPlugin:
    """A validated, locally installed Agent Plugins v1 package."""

    name: str
    root: Path
    version: str
    description: str
    repository: str
    display_name: str
    category: str
    accent_color: str | None
    logo: Path | None
    permissions: tuple[str, ...]
    install_command: tuple[str, ...]


@dataclass(frozen=True)
class AgentPluginState:
    """Runtime state for one discovered Agent Plugin."""

    plugin: AgentPlugin
    mcp_servers: tuple[str, ...]
    enabled: bool
    setup_required: bool


def _discover_agent_plugins(workspace: Path) -> list[AgentPlugin]:
    """Return installed packages found under ``<workspace>/plugins/*``."""
    workspace = workspace.expanduser().resolve()
    root = _contained_directory(workspace / "plugins", workspace)
    if root is None:
        return []
    try:
        candidates = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        logger.warning("Could not inspect Agent Plugins directory: {}", exc)
        return []

    plugins: list[AgentPlugin] = []
    for candidate in candidates:
        plugin_root = _contained_directory(candidate, root)
        if plugin_root is None:
            continue
        plugin = _load_manifest(plugin_root)
        if plugin is not None:
            plugins.append(plugin)
    return plugins


def enabled_agent_plugin_skills(workspace: Path) -> list[AgentPluginSkill]:
    """Return skills from plugins the user has explicitly enabled."""
    skills: list[AgentPluginSkill] = []
    for plugin in _discover_agent_plugins(workspace):
        if _enabled(workspace, plugin.name):
            skills.extend(_discover_plugin_skills(plugin.name, plugin.root))
    return skills


def _load_manifest(plugin_root: Path) -> AgentPlugin | None:
    payload = _read_object(plugin_root / "plugin.json", plugin_root)
    if payload is None:
        return None
    if payload.get("$schema") != AGENT_PLUGIN_SCHEMA:
        return None
    name = payload.get("name")
    if (
        not isinstance(name, str)
        or len(name) > 64
        or _PLUGIN_NAME.fullmatch(name) is None
    ):
        logger.warning("Ignoring Agent Plugin manifest in '{}': invalid name", plugin_root)
        return None
    extension = payload.get("extensions")
    extension_payload = cast(dict[str, object], extension) if isinstance(extension, dict) else {}
    nanobot_value = extension_payload.get("dev.nanobot")
    nanobot = cast(dict[str, object], nanobot_value) if isinstance(nanobot_value, dict) else {}
    return AgentPlugin(
        name=name,
        root=plugin_root,
        version=_string(payload.get("version")),
        description=_string(payload.get("description")),
        repository=_string(payload.get("repository")),
        display_name=_string(nanobot.get("displayName")) or name,
        category=_string(nanobot.get("category")) or "Plugin",
        accent_color=_accent_color(nanobot.get("accentColor")),
        logo=_plugin_logo(nanobot.get("logo"), plugin_root),
        permissions=_string_tuple(nanobot.get("permissions")),
        install_command=_install_command(nanobot.get("installCommand"), plugin_root),
    )


def agent_plugin_mcp_servers(
    workspace: Path,
    configured: dict[str, MCPServerConfig] | None = None,
) -> dict[str, MCPServerConfig]:
    """Merge explicitly enabled plugin MCP servers with user configuration.

    User configuration wins on the unlikely event of a namespaced collision.
    """
    servers: dict[str, MCPServerConfig] = {}
    for plugin in _discover_agent_plugins(workspace):
        if not _enabled(workspace, plugin.name):
            continue
        plugin_servers = _plugin_mcp_servers(workspace, plugin)
        for name, server in plugin_servers.items():
            host_name = plugin.name if len(plugin_servers) == 1 else f"{plugin.name}-{name}"
            servers[host_name] = server
    configured = configured or {}
    if collisions := servers.keys() & configured.keys():
        logger.warning("Configured MCP servers override Agent Plugins: {}", ", ".join(sorted(collisions)))
    return servers | configured


def discover_agent_plugin_states(workspace: Path) -> list[AgentPluginState]:
    """Return component and lifecycle state for discovered plugins."""
    return [
        AgentPluginState(
            plugin=plugin,
            mcp_servers=tuple(sorted(_plugin_mcp_servers(workspace, plugin))),
            enabled=_enabled(workspace, plugin.name),
            setup_required=bool(plugin.install_command)
            and _setup_version(workspace, plugin.name) != (plugin.version or "unknown"),
        )
        for plugin in _discover_agent_plugins(workspace)
    ]


def set_agent_plugin_enabled(workspace: Path, name: str, enabled: bool) -> AgentPlugin:
    """Enable or disable one installed plugin."""
    plugin = next((item for item in _discover_agent_plugins(workspace) if item.name == name), None)
    if plugin is None:
        raise ValueError(f"unknown Agent Plugin '{name}'")
    data = _plugin_data_dir(workspace, plugin.name, create=True)
    with FileLock(str(data / ".state.lock"), timeout=_SETUP_TIMEOUT_SECONDS + 10):
        if enabled:
            if plugin.install_command and _setup_version(workspace, plugin.name) != (
                plugin.version or "unknown"
            ):
                _run_install(plugin, data)
                _write_state(data / "setup-version", plugin.version or "unknown")
            _write_state(data / "enabled", "1")
        else:
            (data / "enabled").unlink(missing_ok=True)
    return plugin


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    items = cast(list[object], value) if isinstance(value, list) else []
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())


def _accent_color(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else None


def _plugin_logo(value: object, plugin_root: Path) -> Path | None:
    """Resolve nanobot's optional packaged logo extension."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("./"):
        logger.warning("Ignoring invalid Agent Plugin logo in '{}'", plugin_root)
        return None
    logo = _contained_file(plugin_root / value[2:], plugin_root)
    try:
        data = logo.read_bytes() if logo is not None else b""
        suffix = logo.suffix.lower() if logo is not None else ""
        valid = (
            suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n")
            or suffix in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8\xff")
            or suffix == ".webp" and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
        )
        if valid and len(data) <= _MAX_LOGO_BYTES:
            return logo
    except OSError:
        pass
    logger.warning("Ignoring invalid Agent Plugin logo in '{}'", plugin_root)
    return None


def _install_command(value: object, plugin_root: Path) -> tuple[str, ...]:
    """Validate nanobot's optional, shell-free setup command extension."""
    if not isinstance(value, list):
        return ()
    items = cast(list[object], value)
    if not 1 <= len(items) <= 32 or not all(
        isinstance(item, str) and 0 < len(item) <= 4096 for item in items
    ):
        return ()
    command = cast(str, items[0])
    if not command.startswith("./"):
        logger.warning("Ignoring non-relative Agent Plugin installCommand in '{}'", plugin_root)
        return ()
    executable = _contained_file(plugin_root / command[2:], plugin_root)
    if executable is None:
        logger.warning("Ignoring invalid Agent Plugin installCommand in '{}'", plugin_root)
        return ()
    return (str(executable), *(cast(str, item) for item in items[1:]))


def _plugin_mcp_servers(workspace: Path, plugin: AgentPlugin) -> dict[str, MCPServerConfig]:
    payload = _read_object(plugin.root / "mcp.json", plugin.root)
    if payload is None:
        return {}
    raw_servers = payload.get("mcpServers")
    if payload.get("$schema") != AGENT_PLUGIN_MCP_SCHEMA or not isinstance(raw_servers, dict):
        logger.warning("Ignoring invalid MCP component for Agent Plugin '{}'", plugin.name)
        return {}

    data = _plugin_data_dir(workspace, plugin.name, create=True)
    servers: dict[str, MCPServerConfig] = {}
    for name, raw in cast(dict[str, object], raw_servers).items():
        if not name or len(name) > 128 or any(ord(char) < 32 for char in name):
            logger.warning("Ignoring invalid MCP server name in Agent Plugin '{}'", plugin.name)
            continue
        server = _plugin_mcp_server(raw, plugin.root, data)
        if server is None:
            logger.warning("Ignoring invalid MCP server '{}' in Agent Plugin '{}'", name, plugin.name)
            continue
        servers[name] = server
    return servers


def _plugin_mcp_server(raw: object, root: Path, data: Path) -> MCPServerConfig | None:
    if not isinstance(raw, dict):
        return None
    payload = cast(dict[str, object], raw)
    if payload.keys() - _MCP_SERVER_FIELDS:
        return None
    try:
        server = MCPServerConfig.model_validate(payload)
    except ValidationError:
        return None
    command = _stdio_command(server.command, root)
    cwd = _stdio_cwd(payload.get("cwd"), root, data)
    if server.type != "stdio" or command is None or cwd is None:
        return None
    if {"PLUGIN_ROOT", "PLUGIN_DATA"} & server.env.keys():
        return None
    return server.model_copy(
        update={
            "command": command,
            "args": [_expand(item, root, data) for item in server.args],
            "env": {
                **{key: _expand(value, root, data) for key, value in server.env.items()},
                "PLUGIN_ROOT": str(root),
                "PLUGIN_DATA": str(data),
            },
            "cwd": str(cwd),
        }
    )


def _stdio_command(value: object, root: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("./"):
        executable = _contained_file(root / value[2:], root)
        return str(executable) if executable is not None else None
    if any(char.isspace() for char in value) or "/" in value or "\\" in value:
        return None
    return value


def _stdio_cwd(value: object, root: Path, data: Path) -> Path | None:
    if value is None:
        return root
    if not isinstance(value, str):
        return None
    if value.startswith("./"):
        return _contained_directory(root / value[2:], root)
    for placeholder, base in (("${PLUGIN_ROOT}", root), ("${PLUGIN_DATA}", data)):
        if value == placeholder or value.startswith(f"{placeholder}/"):
            relative = value[len(placeholder):].lstrip("/")
            candidate = (base / relative).resolve()
            if not candidate.is_relative_to(base):
                return None
            if base == data:
                candidate.mkdir(parents=True, exist_ok=True)
                candidate.chmod(0o700)
            return candidate if candidate.is_dir() else None
    return None


def _expand(value: str, root: Path, data: Path) -> str:
    return value.replace("${PLUGIN_ROOT}", str(root)).replace("${PLUGIN_DATA}", str(data))


def _plugin_data_dir(workspace: Path, name: str, *, create: bool) -> Path:
    workspace_id = sha256(str(workspace.expanduser().resolve()).encode()).hexdigest()[:12]
    config_root = get_config_path().expanduser().resolve().parent
    plugin_root = _private_directory(config_root / "plugin-data", config_root, create=create)
    state_root = _private_directory(plugin_root / workspace_id, config_root, create=create)
    data = state_root / name
    return _private_directory(data, state_root, create=True) if create else data


def _private_directory(path: Path, root: Path, *, create: bool) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        resolved = path.resolve(strict=create)
    except OSError as exc:
        raise RuntimeError("Agent Plugin data directory is unavailable") from exc
    if not resolved.is_relative_to(root):
        raise RuntimeError("Agent Plugin data directory escapes its parent")
    if create:
        resolved.chmod(0o700)
    return resolved


def _enabled(workspace: Path, name: str) -> bool:
    return (_plugin_data_dir(workspace, name, create=False) / "enabled").is_file()


def _setup_version(workspace: Path, name: str) -> str:
    try:
        return (_plugin_data_dir(workspace, name, create=False) / "setup-version").read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, UnicodeError):
        return ""


def _write_state(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def _run_install(plugin: AgentPlugin, data: Path) -> None:
    env = {
        **{key: value for key in _SETUP_ENV if (value := os.environ.get(key)) is not None},
        "PLUGIN_ROOT": str(plugin.root),
        "PLUGIN_DATA": str(data),
    }
    try:
        result = subprocess.run(
            plugin.install_command,
            cwd=plugin.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=_SETUP_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{plugin.display_name} setup timed out") from exc
    if result.returncode:
        output = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(output or f"{plugin.display_name} setup failed")


def _discover_plugin_skills(plugin_name: str, plugin_root: Path) -> list[AgentPluginSkill]:
    skills_root = _contained_directory(plugin_root / "skills", plugin_root)
    if skills_root is None:
        return []

    try:
        candidates = sorted(skills_root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        logger.warning("Could not inspect Agent Plugin '{}' skills: {}", plugin_name, exc)
        return []

    skills: list[AgentPluginSkill] = []
    for candidate in candidates:
        skill_root = _contained_directory(candidate, skills_root)
        if skill_root is None:
            continue
        skill_file = _contained_file(skill_root / "SKILL.md", plugin_root)
        if skill_file is None or not _valid_skill(skill_file, candidate.name, plugin_name):
            continue
        skills.append(
            AgentPluginSkill(name=candidate.name, path=skill_file, plugin=plugin_name)
        )
    return skills


def _valid_skill(path: Path, directory_name: str, plugin_name: str) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    metadata = parse_skill_metadata(content)
    if metadata is None:
        logger.warning("Ignoring Agent Plugin '{}' skill '{}': invalid frontmatter", plugin_name, directory_name)
        return False
    if not valid_skill_metadata(metadata, directory_name):
        logger.warning("Ignoring Agent Plugin '{}' skill '{}': invalid metadata", plugin_name, directory_name)
        return False
    return True


def _contained_directory(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_dir() and resolved.is_relative_to(root) else None


def _read_object(path: Path, root: Path) -> dict[str, object] | None:
    contained = _contained_file(path, root)
    if contained is None:
        return None
    try:
        value = cast(object, json.loads(contained.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid Agent Plugin component '{}': {}", contained, exc)
        return None
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _contained_file(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_file() and resolved.is_relative_to(root) else None
