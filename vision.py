from dataclasses import asdict, dataclass
import gc
import os
import time


MODEL_ALIASES = {
    "qwen": "Qwen/Qwen2.5-VL-3B-Instruct",
    "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    inference_seconds: float
    prompt_characters: int = 0
    visual_input_present: bool = False
    preprocess_seconds: float = 0.0
    time_to_first_token_seconds: float = 0.0
    decode_seconds: float = 0.0
    total_call_seconds: float = 0.0
    prefill_stalled: bool = False
    cuda_free_memory_mb_before: float = 0.0
    cuda_free_memory_mb_after: float = 0.0
    cuda_total_memory_mb: float = 0.0
    cuda_memory_allocated_mb_before: float = 0.0
    cuda_memory_reserved_mb_before: float = 0.0
    cuda_memory_allocated_mb: float = 0.0
    cuda_memory_reserved_mb: float = 0.0
    cuda_free_memory_mb_after_cleanup: float = 0.0
    cuda_memory_allocated_mb_after_cleanup: float = 0.0
    cuda_memory_reserved_mb_after_cleanup: float = 0.0
    generation_index: int = 0
    max_new_tokens_requested: int = 0
    output_hit_token_limit: bool = False
    generation_timed_out: bool = False
    generation_soft_limit_exceeded: bool = False

    def metrics(self):
        return asdict(self)


class VisionLanguageModel:
    def __init__(
        self,
        model_name,
        use_4bit=True,
        device_map="auto",
        max_new_tokens=256,
        min_pixels=None,
        max_pixels=None,
        clear_cuda_cache_between_calls=False,
        generation_max_time_seconds=None,
        four_bit_compute_dtype="auto",
        log_cuda_memory=True,
        cuda_allocator_config=None,
        prefill_stall_threshold_seconds=15.0,
    ):
        self.model_name = MODEL_ALIASES.get(model_name.lower(), model_name)
        self.use_4bit = use_4bit
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.clear_cuda_cache_between_calls = clear_cuda_cache_between_calls
        self.prefill_stall_threshold_seconds = float(prefill_stall_threshold_seconds)
        self.generation_max_time_seconds = (
            float(generation_max_time_seconds)
            if generation_max_time_seconds is not None
            else None
        )
        self.four_bit_compute_dtype = str(
            four_bit_compute_dtype or "auto"
        ).strip().lower()
        self.log_cuda_memory = bool(log_cuda_memory)
        requested_allocator_config = str(
            cuda_allocator_config or ""
        ).strip()
        self.cuda_allocator_config_requested = (
            requested_allocator_config or None
        )
        existing_allocator_config = os.getenv("PYTORCH_CUDA_ALLOC_CONF")
        self.cuda_allocator_config_applied = False
        if requested_allocator_config and not existing_allocator_config:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = requested_allocator_config
            existing_allocator_config = requested_allocator_config
            self.cuda_allocator_config_applied = True
        elif (
            requested_allocator_config
            and existing_allocator_config == requested_allocator_config
        ):
            self.cuda_allocator_config_applied = True
        self.cuda_allocator_config_effective = existing_allocator_config
        self.processor = None
        self.model = None
        self._torch = None
        self._generation_index = 0
        self._resolved_4bit_compute_dtype = None

    @staticmethod
    def _resolve_4bit_compute_dtype(torch_module, requested):
        normalized = str(requested or "auto").strip().lower()
        aliases = {
            "auto": "auto",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp16": "float16",
            "float16": "float16",
            "fp32": "float32",
            "float32": "float32",
        }
        if normalized not in aliases:
            raise ValueError(
                "four_bit_compute_dtype must be one of auto, bfloat16, "
                "float16, or float32."
            )
        normalized = aliases[normalized]

        cuda = getattr(torch_module, "cuda", None)
        supports_bf16 = False
        if cuda is not None and getattr(cuda, "is_available", lambda: False)():
            checker = getattr(cuda, "is_bf16_supported", None)
            if callable(checker):
                try:
                    supports_bf16 = bool(checker())
                except RuntimeError:
                    supports_bf16 = False

        if normalized == "auto":
            normalized = "bfloat16" if supports_bf16 else "float32"
        elif normalized == "bfloat16" and not supports_bf16:
            normalized = "float32"

        dtype = getattr(torch_module, normalized, None)
        if dtype is None:
            raise RuntimeError(
                f"PyTorch does not expose the requested compute dtype: {normalized}."
            )
        return normalized, dtype

    def _cuda_memory_snapshot(self):
        empty = {
            "free_mb": 0.0,
            "total_mb": 0.0,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
        }
        if (
            not self.log_cuda_memory
            or self._torch is None
            or not self._torch.cuda.is_available()
        ):
            return empty
        try:
            free_bytes, total_bytes = self._torch.cuda.mem_get_info()
            return {
                "free_mb": round(free_bytes / (1024 ** 2), 3),
                "total_mb": round(total_bytes / (1024 ** 2), 3),
                "allocated_mb": round(
                    self._torch.cuda.memory_allocated() / (1024 ** 2), 3
                ),
                "reserved_mb": round(
                    self._torch.cuda.memory_reserved() / (1024 ** 2), 3
                ),
            }
        except RuntimeError:
            return empty

    def load(self):
        if self.model is not None:
            return self

        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig
            try:
                from transformers import AutoModelForImageTextToText as AutoVLM
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoVLM
        except ImportError as error:
            raise RuntimeError(
                "The VLM backend requires PyTorch and a recent Transformers "
                "installation. See requirements.txt."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. These experiments are configured for "
                "local GPU inference."
            )

        self._torch = torch
        compute_dtype = None
        if self.use_4bit:
            (
                self._resolved_4bit_compute_dtype,
                compute_dtype,
            ) = self._resolve_4bit_compute_dtype(
                torch,
                self.four_bit_compute_dtype,
            )
        token = os.getenv("HF_TOKEN") or os.getenv("HGF_TOKEN") or None
        processor_options = {"token": token}
        if self.min_pixels is not None:
            processor_options["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_options["max_pixels"] = self.max_pixels

        print(f"Loading {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            **processor_options,
        )

        model_options = {
            "device_map": self.device_map,
            "torch_dtype": "auto",
            "token": token,
            "trust_remote_code": True,
        }
        if self.use_4bit:
            print(
                "4-bit compute dtype: "
                f"{self._resolved_4bit_compute_dtype} "
                f"(requested {self.four_bit_compute_dtype})."
            )
            model_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoVLM.from_pretrained(self.model_name, **model_options)
        self.model.eval()
        print(f"Loaded {self.model_name}.")
        return self

    def generate(self, prompt, image=None, max_new_tokens=None):
        self.load()
        self._generation_index += 1
        request_started = time.perf_counter()
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            template_content = []
            if image is not None:
                template_content.append({"type": "image"})
            template_content.append({"type": "text", "text": prompt})
            template_messages = [{"role": "user", "content": template_content}]
            rendered_prompt = self.processor.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=rendered_prompt,
                images=[image] if image is not None else None,
                return_tensors="pt",
            )

        input_device = next(self.model.parameters()).device
        inputs = inputs.to(input_device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        token_limit = int(max_new_tokens or self.max_new_tokens)
        preprocess_seconds = time.perf_counter() - request_started
        memory_before = self._cuda_memory_snapshot()
        started = time.perf_counter()
        from transformers import StoppingCriteria, StoppingCriteriaList

        class FirstTokenTimer(StoppingCriteria):
            def __init__(self, started_at):
                self.started_at = started_at
                self.elapsed = None

            def __call__(self, input_ids, scores, **kwargs):
                if self.elapsed is None:
                    self.elapsed = time.perf_counter() - self.started_at
                return False

        first_token_timer = FirstTokenTimer(started)
        generation_options = {
            "max_new_tokens": token_limit,
            "do_sample": False,
            "stopping_criteria": StoppingCriteriaList([first_token_timer]),
        }
        if self.generation_max_time_seconds is not None:
            generation_options["max_time"] = self.generation_max_time_seconds
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_options,
            )
        elapsed = time.perf_counter() - started
        memory_after = self._cuda_memory_snapshot()

        generated_ids = output_ids[:, input_tokens:]
        output_tokens = int(generated_ids.shape[-1])
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        soft_limit_exceeded = (
            self.generation_max_time_seconds is not None
            and elapsed >= self.generation_max_time_seconds
        )
        time_to_first_token = first_token_timer.elapsed or 0.0
        del inputs, output_ids, generated_ids
        if self.clear_cuda_cache_between_calls and self._torch.cuda.is_available():
            gc.collect()
            self._torch.cuda.empty_cache()
        memory_after_cleanup = self._cuda_memory_snapshot()
        generation_result = GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            inference_seconds=round(elapsed, 5),
            prompt_characters=len(prompt or ""),
            visual_input_present=image is not None,
            preprocess_seconds=round(preprocess_seconds, 5),
            time_to_first_token_seconds=round(time_to_first_token, 5),
            decode_seconds=round(max(0.0, elapsed - time_to_first_token), 5),
            total_call_seconds=round(preprocess_seconds + elapsed, 5),
            prefill_stalled=time_to_first_token >= self.prefill_stall_threshold_seconds,
            cuda_free_memory_mb_before=memory_before["free_mb"],
            cuda_free_memory_mb_after=memory_after["free_mb"],
            cuda_total_memory_mb=memory_after["total_mb"],
            cuda_memory_allocated_mb_before=memory_before["allocated_mb"],
            cuda_memory_reserved_mb_before=memory_before["reserved_mb"],
            cuda_memory_allocated_mb=memory_after["allocated_mb"],
            cuda_memory_reserved_mb=memory_after["reserved_mb"],
            cuda_free_memory_mb_after_cleanup=memory_after_cleanup["free_mb"],
            cuda_memory_allocated_mb_after_cleanup=(
                memory_after_cleanup["allocated_mb"]
            ),
            cuda_memory_reserved_mb_after_cleanup=(
                memory_after_cleanup["reserved_mb"]
            ),
            generation_index=self._generation_index,
            max_new_tokens_requested=token_limit,
            output_hit_token_limit=output_tokens >= token_limit,
            generation_timed_out=soft_limit_exceeded,
            generation_soft_limit_exceeded=soft_limit_exceeded,
        )
        return generation_result

    def unload(self):
        self.model = None
        self.processor = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def configuration(self):
        return {
            "model_name": self.model_name,
            "use_4bit": self.use_4bit,
            "device_map": self.device_map,
            "max_new_tokens": self.max_new_tokens,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "clear_cuda_cache_between_calls": self.clear_cuda_cache_between_calls,
            "generation_max_time_seconds": self.generation_max_time_seconds,
            "four_bit_compute_dtype_requested": self.four_bit_compute_dtype,
            "four_bit_compute_dtype_resolved": self._resolved_4bit_compute_dtype,
            "log_cuda_memory": self.log_cuda_memory,
            "cuda_allocator_config_requested": self.cuda_allocator_config_requested,
            "cuda_allocator_config_effective": self.cuda_allocator_config_effective,
            "cuda_allocator_config_applied": self.cuda_allocator_config_applied,
            "loaded_dtype": (
                str(getattr(self.model, "dtype", None))
                if self.model is not None
                else None
            ),
        }


VisionModel = VisionLanguageModel
