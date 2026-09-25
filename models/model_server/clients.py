from __future__ import annotations

from models.pipeline.model_client import ModelClient

from .baichuan_omni.client import BaichuanOmni15Client
from .gemini_2_5_flash.client import Gemini25FlashClient
from .gemini_2_5_pro.client import Gemini25ProClient
from .gemini_3_flash_preview.client import Gemini3FlashPreviewClient
from .gemini_3_pro_preview.client import Gemini3ProPreviewClient
from .gpt4o.client import GPT4oClient
from .ming.client import MingClient
from .minicpmo_4_5.client import MiniCPMO45Client
from .miniomni_2.client import MiniOmni2Client
from .omnivinci.client import OmniVinciClient
from .qwen2_5_omni.client import Qwen25OmniClient
from .qwen3_omni.client import Qwen3OmniClient
from .qwen3_omni_thinking.client import Qwen3OmniThinkingClient
from .sglang_omni.client import SGLangOmniClient
from .vita.client import Vita15Client

CLIENTS: dict[str, type[ModelClient]] = {
    "gpt4o": GPT4oClient,
    "gemini_2_5_flash": Gemini25FlashClient,
    "gemini_2_5_pro": Gemini25ProClient,
    "gemini_3_flash_preview": Gemini3FlashPreviewClient,
    "gemini_3_pro_preview": Gemini3ProPreviewClient,
    "qwen3_omni": Qwen3OmniClient,
    "qwen3_omni_thinking": Qwen3OmniThinkingClient,
    "qwen2_5_omni": Qwen25OmniClient,
    "miniomni_2": MiniOmni2Client,
    "minicpmo_4_5": MiniCPMO45Client,
    "omnivinci": OmniVinciClient,
    "vita_1_5": Vita15Client,
    "baichuan_omni_1_5": BaichuanOmni15Client,
    "ming": MingClient,
    "sglang_omni": SGLangOmniClient,
}


def create_client(model_name: str) -> ModelClient:
    try:
        return CLIENTS[model_name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc
