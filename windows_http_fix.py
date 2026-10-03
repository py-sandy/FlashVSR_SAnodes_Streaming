"""Use aiohttp's buffered file transfer on Windows instead of native sendfile."""

import logging
import os
import sys


def enable_windows_file_transfer_fallback():
    if sys.platform != "win32":
        return False
    # The environment setting covers future imports. ComfyUI usually imported
    # aiohttp already, so also update the flag read by FileResponse._sendfile.
    from aiohttp import web_fileresponse
    os.environ["AIOHTTP_NOSENDFILE"] = "1"
    web_fileresponse.NOSENDFILE = True
    logging.getLogger("FlashVSR-SAnodes").info(
        "[SAnodes] Windows HTTP file transfer: buffered fallback enabled "
        "(native sendfile disabled to avoid WinError 87)."
    )
    return True

