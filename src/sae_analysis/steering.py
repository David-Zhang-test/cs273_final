class SaeSteerer:
    def __init__(self, sae_model):
        """
        Initializes the steerer with an SAE model that has decoder weights.
        """
        self.sae = sae_model

    def steering_hook_function(self, activations, hook, target_feature_idx, steering_strength):
        """
        Intercepts the residual stream, decodes the target feature direction 
        from the SAE, and adds it to the model's activations.
        """
        # The SAE decoder weights (W_dec) map the sparse feature back to the dense residual stream
        feature_direction = self.sae.W_dec[target_feature_idx, :]
        
        # We apply the steering vector ONLY at the final token position of the prompt
        # activations shape: [batch, seq_len, d_model]
        activations[0, -1, :] += (feature_direction * steering_strength)
        
        return activations

    def test_steering(self, model_runner, prompt_text, target_layer, feature_idx, strength=50.0):
        """
        Runs generation with the steering vector applied on a specific model runner.
        Use a positive strength to FORCE the behavior, and a negative to SUPPRESS it.
        """
        hook_name = f"blocks.{target_layer}.hook_resid_post"
        
        # Format prompt
        chat_format = [{"role": "user", "content": prompt_text}]
        formatted_prompt = model_runner.model.tokenizer.apply_chat_template(chat_format, tokenize=False)
        tokens = model_runner.model.to_tokens(formatted_prompt)
        
        # Create the hook with our specific feature and strength
        hook_fn = lambda acts, hook: self.steering_hook_function(acts, hook, feature_idx, strength)
        
        # Run generation with the hook attached
        with model_runner.model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            steered_tokens = model_runner.model.generate(tokens, max_new_tokens=150, temperature=0.0)
            
        steered_response = model_runner.model.tokenizer.decode(steered_tokens[0][tokens.shape[1]:])
        return steered_response
