import torch
import csv
import os
import gc
from transformer_lens import HookedTransformer
from tqdm import tqdm


def resolve_torch_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def resolve_fallback_dtype(current_dtype_name):
    if current_dtype_name == "float16":
        return "bfloat16"
    if current_dtype_name == "bfloat16":
        return "float32"
    return None

class ModelRunner:
    def __init__(
        self,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        device="cuda",
        model_dtype="float16",
    ):
        """
        Initializes the model runner.
        Supported models include:
        - "meta-llama/Meta-Llama-3.1-8B-Instruct"
        - "google/gemma-3-12b-it" 
        """
        self.model_name = model_name
        self.device = device
        self.model_dtype = model_dtype
        HF_TOKEN = os.getenv("HF_TOKEN")
        assert HF_TOKEN, "HF_TOKEN environment variable is required to load Hugging Face models in this environment."

        self.model = self._load_model(dtype_name=self.model_dtype)

    def _load_model(self, dtype_name):
        dtype = resolve_torch_dtype(dtype_name)
        return HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=dtype,
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
            # Match any residual stream block
            def hook_filter(name):
                return name.startswith("blocks.") and name.endswith(".hook_resid_post")
        else:
            if isinstance(target_layers, int):
                target_layers = [target_layers]
            hook_names = {f"blocks.{layer}.hook_resid_post" for layer in target_layers}
            def hook_filter(name):
                return name in hook_names

        # Run forward pass with caching
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=hook_filter
            )
            
        # Generate the actual text response
        generated_tokens = self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            verbose=False,
        )
        response_text = self.model.tokenizer.decode(generated_tokens[0][tokens.shape[1]:])
        
        # Extract the activation vector for the strict final token of the prompt
        # at each targeted layer.
        extracted_activations = {}
        for hook_name, act in cache.items():
            if hook_filter(hook_name):
                # Shape: [batch, sequence_pos, d_model] -> we want [0, -1, :]
                extracted_activations[hook_name] = act[0, -1, :]
                
        return response_text, extracted_activations

    def get_full_residual_cache_and_response(self, prompt_text, max_new_tokens=150):
        """Return response, full per-layer residual stream tensors, and last-token vectors."""
        chat_format = [{"role": "user", "content": prompt_text}]
        formatted_prompt = self.model.tokenizer.apply_chat_template(chat_format, tokenize=False)
        tokens = self.model.to_tokens(formatted_prompt)

        def hook_filter(name):
            return name.startswith("blocks.") and name.endswith(".hook_resid_post")

        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, names_filter=hook_filter)

        generated_tokens = self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            verbose=False,
        )
        response_text = self.model.tokenizer.decode(generated_tokens[0][tokens.shape[1]:])

        full_residual_cache = {}
        last_token_cache = {}
        for hook_name, act in cache.items():
            if hook_filter(hook_name):
                # Store [seq_len, d_model] for this prompt.
                full_residual_cache[hook_name] = act[0].detach().cpu()
                # Store [d_model] last-token vector for quick probing.
                last_token_cache[hook_name] = act[0, -1, :].detach().cpu()

        return response_text, full_residual_cache, last_token_cache, tokens[0].detach().cpu()

    def infer_dataset(
        self,
        input_csv_path,
        response_output_csv,
        states_output_dir,
        max_new_tokens=150,
        num_samples=None,
    ):
        """
        Load prompts from CSV (variations.csv schema), run inference, save responses,
        and save all-layer residual stream activations for each prompt to saved_states.
        """
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
