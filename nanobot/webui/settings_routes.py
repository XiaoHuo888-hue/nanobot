"""Stable WebSocket/HTTP dispatcher for WebUI settings domains."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.agent.tools.image_generation import request_image_generation_reload
from nanobot.agent.tools.mcp import request_mcp_reload
from nanobot.api.runtime import ApiRuntime, api_runtime_paths
from nanobot.bus.queue import MessageBus
from nanobot.channels.registry import load_channel_plugin
from nanobot.channels.validation import validate_channel_config
from nanobot.pairing import approve_code, deny_code, list_pending
from nanobot.webui import settings_capabilities as capability_domain
from nanobot.webui import settings_contracts as contracts
from nanobot.webui import settings_models as model_domain
from nanobot.webui import settings_system as system_domain
from nanobot.webui.cli_apps_api import cli_apps_action, cli_apps_payload
from nanobot.webui.http_utils import is_local_browser_request as _is_local_browser_request
from nanobot.webui.mcp_presets_api import mcp_presets_settings_action
from nanobot.webui.nanobot_features_api import (
    nanobot_feature_instance_target,
    nanobot_features_action,
    nanobot_features_payload,
)
from nanobot.webui.settings_api import (
    WebUISettingsError,
    complete_oauth_provider,
    create_model_configuration,
    create_provider_settings,
    decorate_settings_payload,
    delete_model_configuration,
    login_oauth_provider,
    logout_oauth_provider,
    migrate_model_configurations,
    provider_models_payload,
    settings_payload,
    settings_usage_payload,
    update_agent_settings,
    update_api_settings,
    update_image_generation_settings,
    update_model_call_order,
    update_model_configuration,
    update_network_safety_settings,
    update_provider_settings,
    update_transcription_settings,
    update_web_search_settings,
)
from nanobot.webui.settings_contracts import (
    QueryParams,
    SettingsRequest,
    SettingsRouteResult,
)
from nanobot.webui.settings_services import WebUISettingsServices
from nanobot.webui.version_check import check_for_update

_WEBUI_MUTATION_PAYLOAD_ATTR = "_nanobot_webui_mutation_payload"
_WEBUI_MUTATION_REQUEST_ATTR = "_nanobot_webui_mutation_request"
_CHANNEL_CONNECT_ACTIONS = frozenset({"start", "poll", "cancel"})
_query_first = contracts.query_first


def _channel_connect_route(path: str) -> tuple[str, str] | None:
    prefix = "/api/settings/channels/"
    if not path.startswith(prefix):
        return None
    parts = path.removeprefix(prefix).split("/")
    if (
        len(parts) != 3
        or parts[1] != "connect"
        or parts[2] not in _CHANNEL_CONNECT_ACTIONS
    ):
        return None
    channel_name = parts[0].strip()
    return (channel_name, parts[2]) if channel_name else None


_MCP_PRESET_ACTIONS_BY_PATH = {
    "/api/settings/mcp-presets/enable": "enable",
    "/api/settings/mcp-presets/remove": "remove",
    "/api/settings/mcp-presets/test": "test",
    "/api/settings/mcp-presets/custom": "custom",
    "/api/settings/mcp-presets/import": "import",
    "/api/settings/mcp-presets/import-cursor": "import-cursor",
    "/api/settings/mcp-presets/tools": "tools",
}

_MODEL_ROUTES = {
    "/api/settings/update": "agent-update",
    "/api/settings/model-configurations/create": "model-create",
    "/api/settings/model-configurations/update": "model-update",
    "/api/settings/model-configurations/delete": "model-delete",
    "/api/settings/model-configurations/migrate": "models-migrate",
    "/api/settings/model-call-order/update": "call-order-update",
    "/api/settings/provider/update": "provider-update",
    "/api/settings/provider/create": "provider-create",
    "/api/settings/provider-models": "provider-models",
    "/api/settings/provider/oauth-login": "oauth-login",
    "/api/settings/provider/oauth-login/complete": "oauth-complete",
    "/api/settings/provider/oauth-logout": "oauth-logout",
}

_CAPABILITY_ROUTES = {
    "/api/settings/web-search/update": "web-search-update",
    "/api/settings/api-service": "api-status",
    "/api/settings/api-service/start": "api-start",
    "/api/settings/api-service/stop": "api-stop",
    "/api/settings/image-generation/update": "image-update",
    "/api/settings/transcription/update": "transcription-update",
    "/api/settings/network-safety/update": "network-update",
}

_SYSTEM_ROUTES = {
    "/api/settings/cli-apps": "cli-list",
    "/api/settings/cli-apps/install": "cli-install",
    "/api/settings/cli-apps/update": "cli-update",
    "/api/settings/cli-apps/uninstall": "cli-uninstall",
    "/api/settings/cli-apps/test": "cli-test",
    "/api/settings/nanobot-features": "features-list",
    "/api/settings/nanobot-features/enable": "features-enable",
    "/api/settings/nanobot-features/disable": "features-disable",
    "/api/settings/channels/validate": "channel-validate",
    "/api/settings/channels/configure": "channel-configure",
    "/api/settings/pairing": "pairing-list",
    "/api/settings/pairing/approve": "pairing-approve",
    "/api/settings/pairing/deny": "pairing-deny",
    "/api/settings/mcp-presets": "mcp-list",
    "/api/settings/version-check": "version-check",
    **{
        path: f"mcp-{action}"
        for path, action in _MCP_PRESET_ACTIONS_BY_PATH.items()
    },
}

_SETTINGS_MUTATION_PATHS = frozenset({
    "/api/settings/update",
    "/api/settings/model-configurations/create",
    "/api/settings/model-configurations/update",
    "/api/settings/model-configurations/delete",
    "/api/settings/model-configurations/migrate",
    "/api/settings/model-call-order/update",
    "/api/settings/provider/update",
    "/api/settings/provider/create",
    "/api/settings/provider/oauth-login",
    "/api/settings/provider/oauth-login/complete",
    "/api/settings/provider/oauth-logout",
    "/api/settings/web-search/update",
    "/api/settings/api-service/start",
    "/api/settings/api-service/stop",
    "/api/settings/image-generation/update",
    "/api/settings/transcription/update",
    "/api/settings/network-safety/update",
    "/api/settings/cli-apps/install",
    "/api/settings/cli-apps/update",
    "/api/settings/cli-apps/uninstall",
    "/api/settings/cli-apps/test",
    "/api/settings/nanobot-features/enable",
    "/api/settings/nanobot-features/disable",
    "/api/settings/channels/validate",
    "/api/settings/channels/configure",
    "/api/settings/pairing/approve",
    "/api/settings/pairing/deny",
    *_MCP_PRESET_ACTIONS_BY_PATH,
})


def _mutation_payload(request: WsRequest) -> dict[str, Any] | None:
    payload = getattr(request, _WEBUI_MUTATION_PAYLOAD_ATTR, None)
    if not isinstance(payload, dict):
        return None
    return cast(dict[str, Any], payload)


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _payload_query(payload: dict[str, Any]) -> QueryParams:
    return {
        key: [_query_value(value)]
        for key, value in payload.items()
        if key
        and key not in {"authorization_response", "channel", "values"}
    }


class WebUISettingsRouter:
    """Authenticate and dispatch settings requests to transport-neutral domains."""

    def __init__(
        self,
        *,
        settings: WebUISettingsServices,
        bus: MessageBus,
        logger: Any,
        check_api_token: Callable[[WsRequest], bool],
        parse_query: Callable[[str], QueryParams],
        json_response: Callable[[dict[str, Any]], Response],
        error_response: Callable[[int, str | None], Response],
        runtime_surface: str,
        runtime_capabilities: dict[str, Any],
        channel_feature_action: Callable[..., Any] | None = None,
        channel_runtime_status: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.bus = bus
        self.logger = logger
        self._check_api_token = check_api_token
        self._parse_query = parse_query
        self._json_response = json_response
        self._error_response = error_response
        self._runtime_surface = runtime_surface
        self._runtime_capabilities = runtime_capabilities
        self._channel_feature_action = channel_feature_action
        self._channel_runtime_status = channel_runtime_status
        self._restart_sections: set[str] = set()
        self._models = model_domain.ModelSettingsHandler(settings, logger)
        self._capabilities = capability_domain.CapabilitySettingsHandler(
            settings,
            logger,
        )
        self._system = system_domain.SystemSettingsHandler(settings, logger)

    async def dispatch(
        self,
        connection: Any,
        request: WsRequest,
        path: str,
    ) -> Response | None:
        if self.is_mutation_path(path) and not getattr(
            request,
            _WEBUI_MUTATION_REQUEST_ATTR,
            False,
        ):
            return self._error_response(
                405,
                "WebUI mutations require an authenticated WebSocket",
            )

        route = self._route(path)
        if route is None:
            return None
        if not self._authorized(request):
            return self._unauthorized()
        if route == ("root", "settings"):
            return self._handle_settings()
        if route == ("root", "usage"):
            return self._handle_settings_usage()

        domain, action = route
        domain_request = self._domain_request(
            connection,
            request,
            needs_local_browser=(
                action in {
                    "api-start",
                    "features-enable",
                    "channel-configure",
                    "channel-connect",
                }
            ),
        )
        if domain == "models":
            result = await self._models.handle(
                action,
                domain_request,
                self._model_operations(),
            )
        elif domain == "capabilities":
            result = await self._capabilities.handle(
                action,
                domain_request,
                self._capability_operations(),
            )
        else:
            channel_connect = _channel_connect_route(path)
            result = await self._system.handle(
                action,
                domain_request,
                self._system_operations(),
                channel_name=(channel_connect[0] if channel_connect else None),
                connect_action=(channel_connect[1] if channel_connect else None),
            )
        return self._render_result(result)

    @staticmethod
    def is_mutation_path(path: str) -> bool:
        return path in _SETTINGS_MUTATION_PATHS or _channel_connect_route(path) is not None

    @staticmethod
    def _route(path: str) -> tuple[str, str] | None:
        if path == "/api/settings":
            return "root", "settings"
        if path == "/api/settings/usage":
            return "root", "usage"
        if action := _MODEL_ROUTES.get(path):
            return "models", action
        if action := _CAPABILITY_ROUTES.get(path):
            return "capabilities", action
        if action := _SYSTEM_ROUTES.get(path):
            return "system", action
        if _channel_connect_route(path) is not None:
            return "system", "channel-connect"
        return None

    def _query(self, request: WsRequest) -> QueryParams:
        payload = _mutation_payload(request)
        if payload is not None:
            return _payload_query(payload)
        return self._parse_query(request.path)

    def _domain_request(
        self,
        connection: Any,
        request: WsRequest,
        *,
        needs_local_browser: bool,
    ) -> SettingsRequest:
        return SettingsRequest(
            query=self._query(request),
            payload=_mutation_payload(request),
            local_browser=(
                _is_local_browser_request(connection, request.headers)
                if needs_local_browser
                else False
            ),
        )

    def _authorized(self, request: WsRequest) -> bool:
        return self._check_api_token(request)

    def _unauthorized(self) -> Response:
        return self._error_response(401, "Unauthorized")

    def _with_restart_state(
        self,
        payload: dict[str, Any],
        *,
        section: str | None = None,
    ) -> dict[str, Any]:
        if section and payload.get("requires_restart"):
            self._restart_sections.add(section)
        sections = sorted(self._restart_sections)
        updated = dict(payload)
        if sections:
            updated["requires_restart"] = True
        return decorate_settings_payload(
            updated,
            surface=self._runtime_surface,
            runtime_capability_overrides=self._runtime_capabilities,
            restart_required_sections=sections,
        )

    def _render_result(self, result: SettingsRouteResult) -> Response:
        if result.error is not None:
            return self._error_response(result.status, result.error)
        assert result.payload is not None
        payload = result.payload
        if result.clear_restart_section:
            self._restart_sections.discard(result.clear_restart_section)
        if result.decorate_restart:
            if result.restart_payload_key:
                nested = payload.get(result.restart_payload_key)
                if isinstance(nested, dict):
                    payload = dict(payload)
                    payload[result.restart_payload_key] = self._with_restart_state(
                        cast(dict[str, Any], nested),
                        section=result.restart_section,
                    )
            else:
                payload = self._with_restart_state(
                    payload,
                    section=result.restart_section,
                )
        return self._json_response(payload)

    def _handle_settings(self) -> Response:
        return self._json_response(
            self._with_restart_state(
                self.settings.read(
                    settings_payload,
                    surface=self._runtime_surface,
                    runtime_capability_overrides=self._runtime_capabilities,
                )
            )
        )

    def _handle_settings_usage(self) -> Response:
        return self._json_response(self.settings.read(settings_usage_payload))

    def _model_operations(self) -> model_domain.ModelSettingsOperations:
        return model_domain.ModelSettingsOperations(
            update_agent=update_agent_settings,
            create_model=create_model_configuration,
            update_model=update_model_configuration,
            delete_model=delete_model_configuration,
            migrate_models=migrate_model_configurations,
            update_call_order=update_model_call_order,
            update_provider=update_provider_settings,
            create_provider=create_provider_settings,
            provider_models=provider_models_payload,
            oauth_login=login_oauth_provider,
            oauth_complete=complete_oauth_provider,
            oauth_logout=logout_oauth_provider,
            apply_image_runtime_change=self._apply_image_generation_runtime_change_result,
        )

    def _capability_operations(
        self,
    ) -> capability_domain.CapabilitySettingsOperations:
        return capability_domain.CapabilitySettingsOperations(
            update_web_search=update_web_search_settings,
            update_api=update_api_settings,
            update_image=update_image_generation_settings,
            update_transcription=update_transcription_settings,
            update_network=update_network_safety_settings,
            nanobot_features_action=nanobot_features_action,
            api_runtime=self._api_runtime,
            reload_image=lambda: request_image_generation_reload(self.bus),
        )

    def _system_operations(self) -> system_domain.SystemSettingsOperations:
        return system_domain.SystemSettingsOperations(
            cli_apps_payload=cli_apps_payload,
            cli_apps_action=cli_apps_action,
            nanobot_features_payload=nanobot_features_payload,
            nanobot_features_action=nanobot_features_action,
            nanobot_feature_instance_target=nanobot_feature_instance_target,
            validate_channel_config=validate_channel_config,
            load_channel_plugin=load_channel_plugin,
            list_pending=list_pending,
            approve_code=approve_code,
            deny_code=deny_code,
            mcp_presets_action=mcp_presets_settings_action,
            reload_mcp=lambda: request_mcp_reload(self.bus),
            check_for_update=check_for_update,
            channel_feature_action=self._channel_feature_action,
            channel_runtime_status=self._channel_runtime_status,
        )

    async def _apply_image_generation_runtime_change_result(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        return await self._capabilities.apply_image_runtime_change(
            payload,
            lambda: request_image_generation_reload(self.bus),
        )

    async def _apply_image_generation_runtime_change(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        updated, restart_cleared = (
            await self._apply_image_generation_runtime_change_result(payload)
        )
        if restart_cleared:
            self._restart_sections.discard("image")
        return updated

    def _parse_mcp_settings_query(self, request: WsRequest) -> QueryParams:
        return self._query(request)

    def _parse_provider_settings_query(self, request: WsRequest) -> QueryParams:
        return self._query(request)

    def _parse_api_service_settings_query(self, request: WsRequest) -> QueryParams:
        payload = _mutation_payload(request)
        if payload is not None:
            api_key = payload.get("api_key")
            if api_key is not None and not isinstance(api_key, str):
                raise WebUISettingsError("API service API key must be a string")
        return self._query(request)

    def _api_runtime(self) -> ApiRuntime:
        return ApiRuntime(paths=api_runtime_paths(self.settings.config.path))

    def _api_service_payload(
        self,
        *,
        last_action: str | None = None,
    ) -> dict[str, Any]:
        return capability_domain.api_service_payload(
            self.settings,
            self._api_runtime(),
            last_action=last_action,
        )

    @staticmethod
    def _masked_secret(value: str) -> str | None:
        return capability_domain.masked_api_secret(value)

    @staticmethod
    def _api_runtime_message(message: str) -> str:
        return capability_domain.api_runtime_message(message)

    def _parse_channel_values(self, request: WsRequest) -> dict[str, Any]:
        return self._system.parse_channel_values(
            SettingsRequest(
                query=self._query(request),
                payload=_mutation_payload(request),
            )
        )

    def _save_channel_config_values(
        self,
        name: str,
        raw_values: dict[str, Any],
        instance_id: str = "default",
    ) -> list[str]:
        return self.settings.config.update(
            lambda config: system_domain.save_channel_config_values(
                config,
                name,
                raw_values,
                instance_id,
                load_channel_plugin=load_channel_plugin,
            )
        )

    _coerce_channel_value = staticmethod(system_domain.coerce_channel_value)
    _assign_channel_config_value = staticmethod(
        system_domain.assign_channel_config_value
    )

    def _nanobot_features_payload(self) -> dict[str, Any]:
        return nanobot_features_payload(config_path=self.settings.config.path)

    def _nanobot_features_action(
        self,
        action: str,
        query: QueryParams,
        *,
        allow_install: bool = True,
    ) -> dict[str, Any]:
        return self.settings.mutate(
            nanobot_features_action,
            action,
            query,
            allow_install=allow_install,
        )

    @staticmethod
    def _feature_runtime_fallback(
        payload: dict[str, Any],
        *,
        message: str,
    ) -> dict[str, Any]:
        return system_domain.SystemSettingsHandler.feature_runtime_fallback(
            payload,
            message=message,
        )

    def _allow_feature_package_install(
        self,
        connection: Any,
        request: WsRequest,
    ) -> bool:
        domain_request = self._domain_request(
            connection,
            request,
            needs_local_browser=True,
        )
        return self._system.allow_feature_package_install(domain_request)
