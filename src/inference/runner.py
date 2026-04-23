import torch
from transformer_lens import HookedTransformer

class ModelRunner:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda"):
        """
        Initializes the model runner.
        Supported models include:
        - "meta-llama/Meta-Llama-3.1-8B-Instruct"
        - "google/gemma-3-12b-it" 
        """
        self.model_name = model_name
        self.device = device
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)

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
        generated_tokens = self.model.generate(tokens, max_new_tokens=max_new_tokens, temperature=0.0)
        response_text = self.model.tokenizer.decode(generated_tokens[0][tokens.shape[1]:])
        
        # Extract the activation vector for the strict final token of the prompt
        # at each targeted layer.
        extracted_activations = {}
        for hook_name, act in cache.items():
            if hook_filter(hook_name):
                # Shape: [batch, sequence_pos, d_model] -> we want [0, -1, :]
                extracted_activations[hook_name] = act[0, -1, :]
                
        return response_text, extracted_activations
