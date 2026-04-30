import csv
import os
import gc
from transformer_lens import HookedTransformer
from tqdm import tqdm
import torch

from transformer_lens import HookedTransformer
from transformer_lens.cache.key_value_cache_entry import TransformerLensKeyValueCacheEntry

from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_torch_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def patch_transformer_lens_kv_cache_dtype():
    if getattr(TransformerLensKeyValueCacheEntry, "_cfg_dtype_patch_applied", False):
        return

    @classmethod
    def init_cache_entry(cls, cfg, device, batch_size=1):
        n_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else cfg.n_heads
        cache_dtype = getattr(cfg, "dtype", torch.get_default_dtype())
        return cls(
            past_keys=torch.empty((batch_size, 0, n_heads, cfg.d_head), device=device, dtype=cache_dtype),
            past_values=torch.empty((batch_size, 0, n_heads, cfg.d_head), device=device, dtype=cache_dtype),
        )

    TransformerLensKeyValueCacheEntry.init_cache_entry = init_cache_entry
    TransformerLensKeyValueCacheEntry._cfg_dtype_patch_applied = True


class ModelRunner:
    def __init__(
        self,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        device="cuda",
        model_dtype="bfloat16",
        n_devices=1,
    ):
        self.model_name = model_name
        self.device = device
        self.model_dtype = model_dtype
        self.n_devices = int(n_devices)
        self.dtype = resolve_torch_dtype(model_dtype)

        hf_token = os.getenv("HF_TOKEN")
        assert hf_token, "HF_TOKEN environment variable is required to load Hugging Face models in this environment."

        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            available = torch.cuda.device_count()
            assert self.n_devices <= available, (
                f"Requested n_devices={self.n_devices}, but only {available} CUDA device(s) are visible."
            )

        # TransformerLens multi-GPU has cross-device issues on this environment.
        # Use HF backend for n_devices > 1; keep TL backend for single-GPU SAE-compatible path.
        self.backend = "hf" if self.n_devices > 1 else "tl"

        if self.backend == "tl":
            patch_transformer_lens_kv_cache_dtype()
            self.model = self._load_tl_model()
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer, self.model = self._load_hf_model_multi_gpu()

        print(f"ModelRunner backend={self.backend}, dtype={self.model_dtype}, n_devices={self.n_devices}")

    def _load_tl_model(self):
        if self.model_dtype in {"float16", "bfloat16"}:
            return HookedTransformer.from_pretrained_no_processing(
                self.model_name,
                device=self.device,
                dtype=self.dtype,
                n_devices=1,
            )

        return HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
            n_devices=1,
        )

    def _reload_with_fallback_dtype(self):
        fallback = resolve_fallback_dtype(self.model_dtype)
        if fallback is None:
            return False

        print(
            f"Encountered dtype mismatch with {self.model_dtype}. "
            f"Reloading model with {fallback}..."
        )
        # Release previous weights before reloading in a different precision.
        try:
            del self.model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model_dtype = fallback
        self.model = self._load_model(dtype_name=self.model_dtype)
        return True

    def get_activations_and_response(self, prompt_text, target_layers="all", max_new_tokens=150):
        """
        Passes a prompt through the model, caches the residual stream,
        and returns the generated text and the cached activations.
        
        target_layers: list of int, an int, or "all". Ex: [16, 20] or "all"
        """
        # Format prompt using the model's specific chat template
        chat_format = [{"role": "user", "content": prompt_text}]
        formatted_prompt = self.model.tokenizer.apply_chat_template(chat_format, tokenize=False)
        
        # Tokenize
        tokens = self.model.to_tokens(formatted_prompt)
        
        # Determine the hooks to match
        if target_layers == "all":
            hook_names = [f"blocks.{i}.hook_resid_post" for i in range(self.model.cfg.n_layers)]
        else:
            if isinstance(target_layers, int):
                target_layers = [target_layers]
            hook_names = [f"blocks.{i}.hook_resid_post" for i in target_layers]

        residual_cache = {}

        def make_hook(name):
            def _hook(act, hook):
                residual_cache[name] = act[0].detach().cpu()
                return act

            return _hook

        fwd_hooks = [(name, make_hook(name)) for name in hook_names]

        with torch.no_grad():
            with self.model.hooks(fwd_hooks=fwd_hooks):
                _ = self.model(tokens, return_type="logits")

        return residual_cache

    def _build_hf_inputs(self, prompt_text):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        # Put inputs on first mapped device.
        first_dev = 0
        if getattr(self.model, "hf_device_map", None):
            first_dev = next(iter(self.model.hf_device_map.values()))
        if isinstance(first_dev, int):
            device = f"cuda:{first_dev}"
        else:
            device = str(first_dev)

        return {k: v.to(device) for k, v in inputs.items()}

    def _collect_hf_residual_cache(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

        # hidden_states: [embeddings, layer1, ..., layerN]
        hidden_states = outputs.hidden_states[1:]
        full_cache = {
            f"blocks.{i}.hook_resid_post": hs[0].detach().cpu() for i, hs in enumerate(hidden_states)
        }
        return full_cache

    def _generate_hf(self, inputs, max_new_tokens):
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)

    def get_activations_and_response(self, prompt_text, target_layers="all", max_new_tokens=150):
        if self.backend == "tl":
            chat_format = [{"role": "user", "content": prompt_text}]
            formatted_prompt = self.tokenizer.apply_chat_template(chat_format, tokenize=False)
            tokens = self.model.to_tokens(formatted_prompt)

            full_cache = self._collect_tl_residual_cache(tokens, target_layers=target_layers)
            generated_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
            response_text = self.tokenizer.decode(generated_tokens[0][tokens.shape[1] :])
            extracted_activations = {k: v[-1, :] for k, v in full_cache.items()}
            return response_text, extracted_activations

        inputs = self._build_hf_inputs(prompt_text)
        full_cache = self._collect_hf_residual_cache(inputs)
        response_text = self._generate_hf(inputs, max_new_tokens=max_new_tokens)

        if target_layers == "all":
            selected = full_cache
        else:
            if isinstance(target_layers, int):
                target_layers = [target_layers]
            wanted = {f"blocks.{i}.hook_resid_post" for i in target_layers}
            selected = {k: v for k, v in full_cache.items() if k in wanted}

        extracted_activations = {k: v[-1, :] for k, v in selected.items()}
        return response_text, extracted_activations

    def get_full_residual_cache_and_response(self, prompt_text, max_new_tokens=150):
        if self.backend == "tl":
            chat_format = [{"role": "user", "content": prompt_text}]
            formatted_prompt = self.tokenizer.apply_chat_template(chat_format, tokenize=False)
            tokens = self.model.to_tokens(formatted_prompt)

            full_residual_cache = self._collect_tl_residual_cache(tokens, target_layers="all")
            generated_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
            response_text = self.tokenizer.decode(generated_tokens[0][tokens.shape[1] :])
            last_token_cache = {k: v[-1, :] for k, v in full_residual_cache.items()}
            return response_text, full_residual_cache, last_token_cache, tokens[0].detach().cpu()

        inputs = self._build_hf_inputs(prompt_text)
        full_residual_cache = self._collect_hf_residual_cache(inputs)
        response_text = self._generate_hf(inputs, max_new_tokens=max_new_tokens)
        last_token_cache = {k: v[-1, :] for k, v in full_residual_cache.items()}
        prompt_tokens = inputs["input_ids"][0].detach().cpu()
        return response_text, full_residual_cache, last_token_cache, prompt_tokens

    def infer_dataset(
        self,
        input_csv_path,
        response_output_csv,
        states_output_dir,
        max_new_tokens=150,
        num_samples=None,
    ):
        rows = []
        with open(input_csv_path, "r", encoding="utf-8", errors="replace") as file_obj:
            reader = csv.DictReader(file_obj)
            for i, row in enumerate(reader):
                if num_samples is not None and i >= num_samples:
                    break
                rows.append(dict(row))

        if not rows:
            print(f"No rows found in {input_csv_path}")
            return

        os.makedirs(os.path.dirname(response_output_csv), exist_ok=True)
        os.makedirs(states_output_dir, exist_ok=True)

        print(f"Running inference on {len(rows)} prompts from {input_csv_path}...")
        for idx, row in enumerate(tqdm(rows)):
            prompt = row.get("variation", row.get("question", ""))
            state_file = os.path.join(states_output_dir, f"prompt_{idx:06d}.pt")

            try:
                response_text, full_residual_cache, last_token_cache, prompt_tokens = self.get_full_residual_cache_and_response(
                    prompt_text=prompt,
                    max_new_tokens=max_new_tokens,
                )
                row["response"] = response_text

                state_payload = {
                    "prompt_id": idx,
                    "question": row.get("question", ""),
                    "category": row.get("category", ""),
                    "variation_type": row.get("variation_type", ""),
                    "variation": prompt,
                    "response": response_text,
                    "prompt_tokens": prompt_tokens,
                    "residual_stream_full": full_residual_cache,
                    "residual_stream_last_token": last_token_cache,
                }
                torch.save(state_payload, state_file)
                row["state_file"] = state_file
            except Exception as exc:
                # Retry once if this is a known precision mismatch.
                if (
                    (
                        "expected scalar type Float but found Half" in str(exc)
                        or "expected scalar type Float but found BFloat16" in str(exc)
                        or "expected scalar type Half but found BFloat16" in str(exc)
                    )
                    and self.model_dtype in {"float16", "bfloat16"}
                    and self._reload_with_fallback_dtype()
                ):
                    try:
                        response_text, full_residual_cache, last_token_cache, prompt_tokens = self.get_full_residual_cache_and_response(
                            prompt_text=prompt,
                            max_new_tokens=max_new_tokens,
                        )
                        row["response"] = response_text
                        state_payload = {
                            "prompt_id": idx,
                            "question": row.get("question", ""),
                            "category": row.get("category", ""),
                            "variation_type": row.get("variation_type", ""),
                            "variation": prompt,
                            "response": response_text,
                            "prompt_tokens": prompt_tokens,
                            "residual_stream_full": full_residual_cache,
                            "residual_stream_last_token": last_token_cache,
                        }
                        torch.save(state_payload, state_file)
                        row["state_file"] = state_file
                        continue
                    except Exception as retry_exc:
                        exc = retry_exc

                row["response"] = f"ERROR: {exc}"
                row["state_file"] = ""

        with open(response_output_csv, "w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved model responses to {response_output_csv}")
