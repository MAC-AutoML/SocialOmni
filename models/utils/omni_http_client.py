from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests


class OmniHttpClient:
    """Minimal HTTP client: upload video + question, return cleaned answer."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.logger = logging.getLogger("models.utils.omni_http_client")

    def _extract_clean_answer(self, raw_answer: str) -> str:
        if not raw_answer:
            return ""

        lines = [line.strip() for line in raw_answer.split("\n") if line.strip()]
        if not lines:
            return raw_answer.strip()

        last_line = lines[-1]
        if last_line[:1].upper() in {"A", "B", "C", "D"} and len(last_line) == 1:
            return last_line.upper()

        return raw_answer.strip()

    def call_api(
        self,
        video_path: str,
        question: str,
        user_prompt: Optional[str] = None,
        use_video: bool = True,
        use_audio: bool = True,
        visual_mask: bool = False,
        max_retries: int = 5,
        retry_delay: float = 3.0,
    ) -> Optional[str]:
        video_name = os.path.basename(video_path)
        self.logger.info("Calling OMNI API: %s", video_name)

        full_question = f"{user_prompt}\n\n{question}" if user_prompt else question

        for attempt in range(1, max_retries + 1):
            try:
                with open(video_path, "rb") as video_file:
                    response = requests.post(
                        f"{self.server_url}/analyze",
                        files={"video": video_file},
                        data={
                            "question": full_question,
                            "use_video": str(use_video).lower(),
                            "use_audio": str(use_audio).lower(),
                            "visual_mask": str(visual_mask).lower(),
                        },
                        timeout=300,
                    )

                if response.status_code == 200:
                    raw_answer = response.json().get("answer", "")
                    clean_answer = self._extract_clean_answer(raw_answer)
                    self.logger.info("OMNI API ok (attempt %s)", attempt)
                    return clean_answer

                self.logger.warning(
                    "OMNI API bad status=%s (attempt %s)",
                    response.status_code,
                    attempt,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("OMNI API error (attempt %s): %s", attempt, exc)

            if attempt < max_retries:
                time.sleep(retry_delay)

        return None
