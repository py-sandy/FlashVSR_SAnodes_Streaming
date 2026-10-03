"""Bounded, overlapping video streaming for the existing FlashVSR nodes."""

from .windows_http_fix import enable_windows_file_transfer_fallback

enable_windows_file_transfer_fallback()

from .stream_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
