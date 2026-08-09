"""Discover portable Agent Plugins from the agent workspace."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import yaml
from loguru import logger

from nanobot.config.loader import get_config_path
from nanobot.config.schema import MCPServerConfig

AGENT_PLUGIN_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
AGENT_PLUGIN_MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"

_PLUGIN_NAME = re.compile(r"^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_SKILL_NAME = re.compile(r"^(?!.*--)[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_SKILL_FRONTMATTER = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?", re.DOTALL)
_MANIFEST_FIELDS = {
    "$schema",
    "name",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
    "extensions",
}
_STRING_FIELDS = {"version", "description", "homepage", "repository", "license"}
_AUTHOR_FIELDS = {"name", "email", "url"}
_MCP_SERVER_FIELDS = {
    "stdio": {"type", "command", "args", "env", "cwd"},
}
_SETUP_ENV = {"HOME", "LANG", "LC_ALL", "LOGNAME", "PATH", "SHELL", "TMPDIR", "USER"}


@dataclass(frozen=True)
class AgentPluginSkill:
    """One skill supplied by a valid Agent Plugins v1 package."""

    name: str
    path: Path
    plugin: str


@dataclass(frozen=True)
class AgentPlugin:
    """A validated Agent Plugins v1 package installed in the workspace."""

    name: str
    root: Path
    version: str
    description: str
    repository: str
    display_name: str
    category: str
    accent_color: str | None
    permissions: tuple[str, ...]
    install_command: tuple[str, ...]


def discover_agent_plugins(workspace: Path) -> list[AgentPlugin]:
    """Return valid packages from ``<workspace>/plugins/*``."""
    workspace = workspace.expanduser().resolve()
    plugins_root = workspace / "plugins"
    if not plugins_root.is_dir():
        return []
    try:
        root = plugins_root.resolve(strict=True)
    except OSError:
        return []
    if not root.is_relative_to(workspace):
        logger.warning("Ignoring Agent Plugins directory outside the workspace")
        return []
    try:
        candidates = sorted(plugins_root.iterdir(), key=lambda path: path.name)
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


def discover_agent_plugin_skills(workspace: Path) -> list[AgentPluginSkill]:
    """Discover direct-child skills under ``<workspace>/plugins/*``.

    Agent Plugins does not prescribe an install location. nanobot uses the
    workspace ``plugins`` directory so packages stay explicit and portable
    with the rest of the agent workspace.
    """
    skills: list[AgentPluginSkill] = []
    for plugin in discover_agent_plugins(workspace):
        skills.extend(_discover_plugin_skills(plugin.name, plugin.root))
    return skills


def _load_manifest(plugin_root: Path) -> AgentPlugin | None:
    manifest = _contained_file(plugin_root / "plugin.json", plugin_root)
    if manifest is None:
        return None
    try:
        value = cast(object, json.loads(manifest.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid Agent Plugin manifest '{}': {}", manifest, exc)
        return None
    if not isinstance(value, dict):
        logger.warning("Ignoring Agent Plugin manifest '{}': expected a JSON object", manifest)
        return None

    payload = cast(dict[str, Any], value)
    if payload.get("$schema") != AGENT_PLUGIN_SCHEMA:
        return None
    name = payload.get("name")
    if (
        not isinstance(name, str)
        or len(name) > 64
        or _PLUGIN_NAME.fullmatch(name) is None
    ):
        logger.warning("Ignoring Agent Plugin manifest '{}': invalid name", manifest)
        return None
    if not _valid_optional_fields(payload):
        logger.warning("Ignoring Agent Plugin manifest '{}': invalid metadata", manifest)
        return None

    for field in payload.keys() - _MANIFEST_FIELDS:
        logger.warning("Ignoring unknown Agent Plugin manifest field '{}' in '{}'", field, manifest)
    if "extensions" in payload and not isinstance(payload["extensions"], dict):
        logger.warning("Ignoring non-object Agent Plugin extensions in '{}'", manifest)
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
    for plugin in discover_agent_plugins(workspace):
        if not _enabled(workspace, plugin.name):
            continue
        plugin_servers = _plugin_mcp_servers(workspace, plugin)
        for name, server in plugin_servers.items():
            host_name = plugin.name if len(plugin_servers) == 1 else f"{plugin.name}-{name}"
            servers[host_name] = server
    for name, server in (configured or {}).items():
        if name in servers:
            logger.warning("Configured MCP server '{}' overrides an Agent Plugin server", name)
        servers[name] = server
    return servers


def agent_plugins_payload(workspace: Path) -> dict[str, Any]:
    """Return installed Agent Plugins for the WebUI Apps surface."""
    plugins: list[dict[str, Any]] = []
    enabled_count = 0
    for plugin in discover_agent_plugins(workspace):
        mcp_servers = sorted(_plugin_mcp_servers(workspace, plugin))
        if not mcp_servers and not plugin.install_command:
            continue
        enabled = _enabled(workspace, plugin.name)
        enabled_count += int(enabled)
        plugins.append(
            {
                "name": plugin.name,
                "display_name": plugin.display_name,
                "version": plugin.version,
                "description": plugin.description,
                "category": plugin.category,
                "repository": plugin.repository,
                "accent_color": plugin.accent_color,
                "permissions": list(plugin.permissions),
                "mcp_servers": mcp_servers,
                "enabled": enabled,
                "setup_required": bool(plugin.install_command)
                and _setup_version(workspace, plugin.name) != (plugin.version or "unknown"),
            }
        )
    return {"plugins": plugins, "enabled_count": enabled_count}


def set_agent_plugin_enabled(workspace: Path, name: str, enabled: bool) -> dict[str, Any]:
    """Enable or disable one installed plugin's executable MCP components."""
    plugin = next((item for item in discover_agent_plugins(workspace) if item.name == name), None)
    if plugin is None:
        raise ValueError(f"unknown Agent Plugin '{name}'")
    data = _plugin_data_dir(workspace, plugin.name, create=True)
    if enabled:
        if plugin.install_command and _setup_version(workspace, plugin.name) != (plugin.version or "unknown"):
            _run_install(plugin, data)
            _write_state(data / "setup-version", plugin.version or "unknown")
        _write_state(data / "enabled", "1")
    else:
        (data / "enabled").unlink(missing_ok=True)
    payload = agent_plugins_payload(workspace)
    payload["last_action"] = {
        "ok": True,
        "message": f"{plugin.display_name} {'enabled' if enabled else 'disabled'}.",
    }
    return payload


def _valid_optional_fields(payload: dict[str, Any]) -> bool:
    if any(field in payload and not isinstance(payload[field], str) for field in _STRING_FIELDS):
        return False
    keywords = payload.get("keywords")
    if "keywords" in payload and (
        not isinstance(keywords, list)
        or not all(isinstance(keyword, str) for keyword in cast(list[object], keywords))
    ):
        return False
    author = payload.get("author")
    if "author" not in payload:
        return True
    if not isinstance(author, dict):
        return False
    author_payload = cast(dict[str, object], author)
    return not (author_payload.keys() - _AUTHOR_FIELDS) and all(
        isinstance(value, str) for value in author_payload.values()
    )


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items = cast(list[object], value)
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())


def _accent_color(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else None


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
    path = _contained_file(plugin.root / "mcp.json", plugin.root)
    if path is None:
        return {}
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid MCP component for Agent Plugin '{}': {}", plugin.name, exc)
        return {}
    if not isinstance(value, dict):
        return {}
    payload = cast(dict[str, Any], value)
    raw_servers = payload.get("mcpServers")
    if (
        payload.keys() != {"$schema", "mcpServers"}
        or payload.get("$schema") != AGENT_PLUGIN_MCP_SCHEMA
        or not isinstance(raw_servers, dict)
    ):
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
    payload = cast(dict[str, Any], raw)
    transport = payload.get("type")
    allowed = _MCP_SERVER_FIELDS.get(transport) if isinstance(transport, str) else None
    if allowed is None or payload.keys() - allowed:
        return None
    if transport == "stdio":
        command = _stdio_command(payload.get("command"), root)
        args = payload.get("args", [])
        env = payload.get("env", {})
        cwd = _stdio_cwd(payload.get("cwd"), root, data)
        if (
            command is None
            or not isinstance(args, list)
            or not all(isinstance(item, str) for item in cast(list[object], args))
            or not isinstance(env, dict)
            or cwd is None
        ):
            return None
        env_payload = cast(dict[object, object], env)
        if any(
            not isinstance(key, str)
            or key in {"PLUGIN_ROOT", "PLUGIN_DATA"}
            or not isinstance(value, str)
            for key, value in env_payload.items()
        ):
            return None
        string_env = cast(dict[str, str], env)
        replacements = {"${PLUGIN_ROOT}": str(root), "${PLUGIN_DATA}": str(data)}
        return MCPServerConfig(
            type="stdio",
            command=command,
            args=[_expand(item, replacements) for item in cast(list[str], args)],
            env={
                **{key: _expand(value, replacements) for key, value in string_env.items()},
                "PLUGIN_ROOT": str(root),
                "PLUGIN_DATA": str(data),
            },
            cwd=str(cwd),
        )

    return None


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


def _expand(value: str, replacements: dict[str, str]) -> str:
    for token, replacement in replacements.items():
        value = value.replace(token, replacement)
    return value


def _plugin_data_dir(workspace: Path, name: str, *, create: bool) -> Path:
    workspace_id = sha256(str(workspace.expanduser().resolve()).encode()).hexdigest()[:12]
    config_root = get_config_path().expanduser().resolve().parent
    plugin_data_root = config_root / "plugin-data"
    if create:
        plugin_data_root.mkdir(parents=True, exist_ok=True)
    try:
        resolved_plugin_data = plugin_data_root.resolve(strict=create)
    except OSError as exc:
        raise RuntimeError("Agent Plugin data root is unavailable") from exc
    if not resolved_plugin_data.is_relative_to(config_root):
        raise RuntimeError("Agent Plugin data root escapes the nanobot config directory")
    state_root = resolved_plugin_data / workspace_id
    if create:
        resolved_plugin_data.chmod(0o700)
        state_root.mkdir(parents=True, exist_ok=True)
    try:
        resolved_state = state_root.resolve(strict=create)
    except OSError as exc:
        raise RuntimeError("Agent Plugin state directory is unavailable") from exc
    if not resolved_state.is_relative_to(config_root):
        raise RuntimeError("Agent Plugin state directory escapes the nanobot config directory")
    if create:
        resolved_state.chmod(0o700)
    data = resolved_state / name
    if create:
        data.mkdir(exist_ok=True)
        resolved_data = data.resolve(strict=True)
        if not resolved_data.is_relative_to(resolved_state):
            raise RuntimeError("Agent Plugin data directory escapes its state directory")
        resolved_data.chmod(0o700)
        return resolved_data
    return data


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
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
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{plugin.display_name} setup timed out") from exc
    if result.returncode:
        output = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(output or f"{plugin.display_name} setup failed")


def _discover_plugin_skills(plugin_name: str, plugin_root: Path) -> list[AgentPluginSkill]:
    skills_root = plugin_root / "skills"
    if not skills_root.exists():
        return []
    resolved_skills_root = _contained_directory(skills_root, plugin_root)
    if resolved_skills_root is None:
        logger.warning("Ignoring invalid skills component in Agent Plugin '{}'", plugin_name)
        return []

    try:
        candidates = sorted(skills_root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        logger.warning("Could not inspect Agent Plugin '{}' skills: {}", plugin_name, exc)
        return []

    skills: list[AgentPluginSkill] = []
    for candidate in candidates:
        skill_root = _contained_directory(candidate, resolved_skills_root)
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
    match = _SKILL_FRONTMATTER.match(content)
    if match is None:
        logger.warning("Ignoring Agent Plugin '{}' skill '{}': invalid frontmatter", plugin_name, directory_name)
        return False
    try:
        metadata = cast(object, yaml.safe_load(match.group(1)))
    except yaml.YAMLError:
        metadata = None
    if not isinstance(metadata, dict):
        logger.warning("Ignoring Agent Plugin '{}' skill '{}': invalid frontmatter", plugin_name, directory_name)
        return False
    payload = cast(dict[object, object], metadata)
    name = payload.get("name")
    description = payload.get("description")
    valid = (
        name == directory_name
        and isinstance(name, str)
        and len(name) <= 64
        and _SKILL_NAME.fullmatch(name) is not None
        and isinstance(description, str)
        and 1 <= len(description.strip()) <= 1024
    )
    if not valid:
        logger.warning("Ignoring Agent Plugin '{}' skill '{}': invalid metadata", plugin_name, directory_name)
    return valid


def _contained_directory(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_dir() and resolved.is_relative_to(root) else None


def _contained_file(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_file() and resolved.is_relative_to(root) else None
