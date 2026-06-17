"""oprim._providers.duix — Duix-Avatar local Docker REST API.

Three-service Docker stack: fun-asr + fish-speech + duix.avatar.
Endpoint: POST 127.0.0.1:8383/easy/submit → poll GET /easy/query?code=<uuid>
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path


class DuixError(Exception):
    """Duix avatar generation failed."""


class DuixSubmitError(DuixError):
    """Submit request to Duix service failed."""


class DuixPollTimeoutError(DuixError):
    """Polling timed out before completion."""


async def submit_and_poll(
    *,
    portrait_image: Path,
    audio_path: Path,
    output_path: Path,
    base_url: str = "http://127.0.0.1:8383",
    poll_interval_s: float = 3.0,
    timeout_s: float = 300.0,
) -> Path:
    """Submit a lip-sync job to Duix-Avatar and poll until complete.

    Args:
        portrait_image: Portrait/video reference file.
        audio_path: Audio file for lip-sync.
        output_path: Destination video file.
        base_url: Duix service base URL.
        poll_interval_s: Seconds between status polls.
        timeout_s: Total timeout.

    Returns:
        output_path on success.

    Raises:
        DuixSubmitError: submit endpoint returned an error.
        DuixPollTimeoutError: job did not complete within timeout_s.
        DuixError: job failed or output could not be downloaded.
    """
    import httpx

    job_code = str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        submit_resp = await client.post(
            f"{base_url}/easy/submit",
            json={
                "audio_url": str(audio_path),
                "video_url": str(portrait_image),
                "code": job_code,
            },
        )
        if submit_resp.status_code != 200:
            raise DuixSubmitError(
                f"Duix submit error {submit_resp.status_code}: {submit_resp.text[:200]}"
            )
        submit_data = submit_resp.json()
        if submit_data.get("code") != 10000:
            raise DuixSubmitError(
                f"Duix submit rejected: {submit_data}"
            )

        # Poll
        elapsed = 0.0
        while elapsed < timeout_s:
            await asyncio.sleep(poll_interval_s)
            elapsed += poll_interval_s

            query_resp = await client.get(
                f"{base_url}/easy/query",
                params={"code": job_code},
            )
            if query_resp.status_code != 200:
                raise DuixError(
                    f"Duix poll error {query_resp.status_code}: {query_resp.text[:200]}"
                )
            qdata = query_resp.json()
            status = qdata.get("status", "")

            if status in ("completed", "success", "done", 2, "2"):
                # Duix returns container-local path in data["result"]
                result_path_str: str | None = (
                    qdata.get("result")
                    or qdata.get("video_url")
                    or qdata.get("url")
                )
                if not result_path_str:
                    raise DuixError(f"No result path in Duix response: {qdata}")
                result_path = Path(result_path_str)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if result_path.exists():
                    import shutil
                    shutil.copy2(result_path, output_path)
                else:
                    # Fallback: try HTTP download if looks like URL
                    if result_path_str.startswith("http"):
                        dl = await client.get(result_path_str)
                        if dl.status_code != 200:
                            raise DuixError(f"Duix video download failed {dl.status_code}")
                        output_path.write_bytes(dl.content)
                    else:
                        raise DuixError(f"Duix result path not found: {result_path_str}")
                return output_path

            if status in ("failed", "error", "-1"):
                raise DuixError(
                    f"Duix job failed: {qdata.get('message', qdata)}"
                )

        raise DuixPollTimeoutError(
            f"Duix job {job_code} did not complete within {timeout_s}s"
        )
