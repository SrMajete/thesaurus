"""Infrastructure adapters — implementations of core ports.

Each module here wraps an external system (Anthropic SDK, filesystem,
Textual TUI, OS environment) behind a port defined in ``core/``.
Adapters depend on core; core never imports from here.
"""
