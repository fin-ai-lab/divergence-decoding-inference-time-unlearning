from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F


class WHP(torch.nn.Module):
    """
    Who's Harry Potter (WHP) model.

    At inference: logits = baseline_logits - alpha * ReLU(reinforced_logits - baseline_logits)

    The ReLU gate ensures we only subtract where the reinforced model's logits
    exceed the baseline's, targeting specifically the knowledge added by finetuning.

    Requires:
      - model_whp_baseline: path to the original/target model
      - model_whp_reinforced: path to the model finetuned on the forget set
      - model_whp_alpha: scaling coefficient (default 1.0)
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.device = model_cfg.get("device", "cuda")
        self.alpha = float(model_cfg.get("model_whp_alpha", 1.0))

        # Initialize tokenizer
        if "muse-bench" in model_cfg.model_whp_baseline:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_whp_baseline)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.special_token_ids = self.tokenizer.all_special_ids

        device_map = {"": self.device}

        # Load baseline (original/target) model
        self.baseline = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_whp_baseline,
            torch_dtype=torch.float16,
            device_map=device_map
        ).eval()

        # Load reinforced (finetuned on forget set) model
        self.reinforced = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_whp_reinforced,
            torch_dtype=torch.float16,
            device_map=device_map
        ).eval()

    def generate(self, input_ids, attention_mask=None, pad_token_id=None, **generation_args):
        max_new_tokens = generation_args.get('max_new_tokens', 200)
        batch_size = input_ids.shape[0]
        device = self.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        past_kv_baseline = None
        past_kv_reinforced = None

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated = input_ids.clone()
        current_input = input_ids
        current_attention_mask = attention_mask.clone()

        for step in range(max_new_tokens):
            if finished.all():
                break

            with torch.no_grad():
                out_baseline = self.baseline(
                    current_input,
                    attention_mask=current_attention_mask,
                    past_key_values=past_kv_baseline,
                    use_cache=True
                )
                out_reinforced = self.reinforced(
                    current_input,
                    attention_mask=current_attention_mask,
                    past_key_values=past_kv_reinforced,
                    use_cache=True
                )

                baseline_logits = out_baseline.logits[:, -1, :]
                reinforced_logits = out_reinforced.logits[:, -1, :]

                past_kv_baseline = out_baseline.past_key_values
                past_kv_reinforced = out_reinforced.past_key_values

                # WHP: baseline - alpha * ReLU(reinforced - baseline)
                logits = baseline_logits - self.alpha * F.relu(reinforced_logits - baseline_logits)

            # Sampling
            if generation_args.get('do_sample', False):
                temperature = generation_args.get('temperature', 1.0)
                if temperature != 1.0:
                    logits = logits / temperature

                top_p = generation_args.get('top_p', None)
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            next_tokens[finished] = pad_token_id if pad_token_id is not None else 0
            generated = torch.cat((generated, next_tokens), dim=1)

            new_attention = torch.ones(batch_size, 1, device=device)
            new_attention[finished] = 0
            current_attention_mask = torch.cat((current_attention_mask, new_attention), dim=1)

            if hasattr(self, 'special_token_ids'):
                for i, token in enumerate(next_tokens.squeeze(1)):
                    if not finished[i] and token.item() in self.special_token_ids:
                        finished[i] = True

            current_input = next_tokens

        return generated

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        """Forward pass with WHP logit combination."""
        device = self.device

        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)

        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, device=device)

        # Handle past_key_values
        past_kv_baseline = None
        past_kv_reinforced = None
        if past_key_values is not None:
            if isinstance(past_key_values, (tuple, list)) and len(past_key_values) == 2:
                past_kv_baseline, past_kv_reinforced = past_key_values
            else:
                past_kv_baseline = past_key_values

        with torch.no_grad():
            out_baseline = self.baseline(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv_baseline,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs
            )

            out_reinforced = self.reinforced(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv_reinforced,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                **kwargs
            )

        # WHP: baseline - alpha * ReLU(reinforced - baseline)
        logits = out_baseline.logits - self.alpha * F.relu(out_reinforced.logits - out_baseline.logits)

        modified_outputs = out_baseline
        modified_outputs.logits = logits

        if use_cache and modified_outputs.past_key_values is not None:
            modified_outputs.past_key_values = (
                out_baseline.past_key_values,
                out_reinforced.past_key_values
            )

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            modified_outputs.loss = loss

        return modified_outputs
