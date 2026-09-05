"""Domain-specific application failures."""


class AgentLabError(Exception):
    """Base class for expected failures."""


class ConfigurationError(AgentLabError):
    """Configuration or required environment is invalid."""


class ModelError(AgentLabError):
    """A model request failed or returned an invalid response."""


class SearchError(AgentLabError):
    """A search request failed or returned invalid data."""


class StateError(AgentLabError):
    """Graph state is incomplete or invalid."""


class IntegrationUnavailableError(AgentLabError):
    """An optional runtime integration is not installed."""


class GraphError(AgentLabError):
    """A graph could not be assembled or executed."""
