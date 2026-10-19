"""Grounded RAG generation with Qwen3-4B-Instruct-2507.

The model is intentionally loaded lazily: starting the API only creates the
generator object, while the first question loads the weights onto the selected
device.  This keeps the existing FastAPI lifecycle lightweight and avoids an
Ollama service dependency.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator

import torch

from src.core.config import settings

logger = logging.getLogger(__name__)


class RAGGenerator:
    """Generate an answer from retrieved evidence using Qwen3-4B-Instruct."""

    def __init__(self, max_context_length: int | None = None) -> None:
        self.max_context_length = max_context_length or settings.RAG_MAX_CONTEXT_CHARS
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-rag")

    def _build_messages(self, query: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Build a Vietnamese, evidence-grounded chat prompt.

        The character cap preserves compatibility with the old generator and
        provides a first safety bound.  A second token-aware bound is enforced
        immediately before generation.
        """
        context_text = "\n\n".join(
            f"[Nguồn {index + 1}]\n{document.get('content', '')}"
            for index, document in enumerate(contexts)
        )[: self.max_context_length]

        system_prompt = f"""Bạn là {settings.BOT_NAME}, một trợ lý trả lời bằng tiếng Việt.

Chỉ sử dụng thông tin trong NGỮ CẢNH. Chỉ trả lời khi NGỮ CẢNH nêu rõ
toàn bộ thông tin mà câu hỏi yêu cầu; một thông tin cùng chủ đề nhưng không
trả lời đúng thuộc tính được hỏi là không đủ căn cứ.

Trước khi trả lời, xác định thông tin cụ thể cần tìm trong câu hỏi, như con số,
thời hạn, điều kiện, đối tượng, bước thực hiện hoặc sự khác biệt. Nếu NGỮ CẢNH
không nêu rõ chính xác thông tin đó, chỉ trả lời đúng: "Tài liệu không đề cập."

Không suy luận từ số thứ tự điều, khoản, mục, danh sách, số thành viên, số năm
hoặc thông tin khác không được nêu rõ là câu trả lời cho câu hỏi.

Chỉ xuất ra câu trả lời trực tiếp; không mở đầu bằng “Theo ngữ cảnh”, không
thêm tiêu đề, Markdown, diễn giải thừa hoặc nhãn nguồn. Với câu hỏi so sánh,
chỉ nêu các đối tượng và điểm khác biệt được hỏi. Với câu hỏi định nghĩa, nêu
đủ định nghĩa khi cần nhưng không thêm thông tin ngoài câu hỏi."""
        user_prompt = f"""NGỮ CẢNH:
{context_text}

CÂU HỎI: {query}

TRẢ LỜI:"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _device_and_dtype(self) -> tuple[str, torch.dtype]:
        requested_device = settings.GENERATOR_DEVICE.lower()
        if requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GENERATOR_DEVICE=cuda nhưng PyTorch không thấy GPU CUDA. "
                    "Hãy chọn máy có GPU hoặc đặt GENERATOR_DEVICE=cpu."
                )
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return requested_device, dtype
        if requested_device == "cpu":
            return "cpu", torch.float32
        raise ValueError("GENERATOR_DEVICE chỉ hỗ trợ 'cuda' (hoặc cuda:N) và 'cpu'.")

    def _load_model_sync(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype = self._device_and_dtype()
        token = settings.HF_TOKEN or None
        logger.info("Loading RAG generator %s on %s", settings.HF_MODEL_NAME, device)
        tokenizer = AutoTokenizer.from_pretrained(settings.HF_MODEL_NAME, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            settings.HF_MODEL_NAME,
            token=token,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        self._model = model

    async def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self._load_model_sync)

    def _prepare_inputs(self, query: str, contexts: list[dict[str, Any]]) -> Any:
        """Tokenize while shrinking only evidence, never the question or policy."""
        assert self._tokenizer is not None
        tokenizer = self._tokenizer
        current_limit = self.max_context_length

        while True:
            original_limit = self.max_context_length
            self.max_context_length = current_limit
            try:
                messages = self._build_messages(query, contexts)
            finally:
                self.max_context_length = original_limit

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if input_ids.shape[-1] <= settings.GENERATOR_MAX_INPUT_TOKENS:
                return input_ids
            if current_limit <= 256:
                raise ValueError(
                    "Câu hỏi và hướng dẫn vượt GENERATOR_MAX_INPUT_TOKENS; "
                    "hãy tăng giới hạn này hoặc rút ngắn câu hỏi."
                )
            current_limit = max(256, current_limit // 2)

    def _generate_sync(
        self, query: str, contexts: list[dict[str, Any]]
    ) -> tuple[str, dict[str, int]]:
        assert self._model is not None
        input_ids = self._prepare_inputs(query, contexts).to(self._model.device)
        attention_mask = torch.ones_like(input_ids, device=self._model.device)
        prompt_tokens = input_ids.shape[-1]

        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        answer = self._tokenizer.decode(
            generated_ids[0, prompt_tokens:], skip_special_tokens=True
        ).strip()
        if not answer:
            raise RuntimeError("Qwen không sinh được câu trả lời.")
        output_tokens = int(generated_ids.shape[-1] - prompt_tokens)
        return answer, {
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": output_tokens,
            "total_tokens": int(prompt_tokens + output_tokens),
        }

    async def generate_with_metrics(
        self, query: str, contexts: list[dict[str, Any]]
    ) -> tuple[str, dict[str, int]]:
        """Generate one answer and return exact tokenizer-level token counts."""
        if not contexts:
            return "Tài liệu không đề cập.", {
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        await self._ensure_model_loaded()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._generate_sync, query, contexts)

    async def generate_stream(
        self, query: str, contexts: list[dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """Return a grounded answer through the existing streaming API.

        Inference runs in a dedicated worker so it does not block FastAPI's
        event loop.  The completed answer is emitted as one chunk; the API
        contract remains an async stream for the UI and cache layers.
        """
        answer, _ = await self.generate_with_metrics(query, contexts)
        yield answer

    async def aclose(self) -> None:
        """Release worker resources during application shutdown."""
        self._executor.shutdown(wait=False, cancel_futures=True)
