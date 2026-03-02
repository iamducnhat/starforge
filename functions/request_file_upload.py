"""Tool: request_file_upload

Simple helper that generates an upload request identifier and
returns human-readable instructions the front-end or user can follow
to attach a file for the assistant to consume.
"""
import uuid
from typing import List, Optional, Dict, Any


def request_file_upload(purpose: str = "general",
                        accepted_types: Optional[List[str]] = None,
                        max_size_bytes: Optional[int] = None,
                        filename: Optional[str] = None) -> Dict[str, Any]:
    """Create a short payload instructing the user to upload a file.

    This function does not perform the upload itself. Instead it returns
    an `upload_id` plus instructions the UI or user can use to attach the
    desired file. The environment that receives the uploaded file should
    record the `upload_id` so the assistant can later reference it.

    Args:
        purpose: Short description of why the file is requested.
        accepted_types: Optional list of MIME types or extensions (e.g. ["image/png"]).
        max_size_bytes: Optional maximum allowed size in bytes.
        filename: Optional suggested filename.

    Returns:
        A dict containing an `upload_id`, the provided constraints and a
        human-readable `message` and `instructions` for attaching the file.
    """
    upload_id = str(uuid.uuid4())
    return {
        "upload_id": upload_id,
        "purpose": purpose,
        "accepted_types": accepted_types or [],
        "max_size_bytes": max_size_bytes,
        "filename": filename,
        "message": (
            "Please upload the requested file and include the provided `upload_id` "
            "in the upload metadata. Once uploaded, call or send the upload metadata "
            "back to the assistant so it can locate the file."
        ),
        "instructions": (
            "Attach the file via the UI/chat upload, and set the metadata field 'upload_id' "
            "to this value. Accepted types and size limits are provided above."
        ),
    }
