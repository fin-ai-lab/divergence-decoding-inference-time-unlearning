"""
GUARD (Generation-time Unlearning via Adaptive Restriction and Detection) model handler.

At inference: classifies prompts as forget/retain using a trained MLP on
penultimate-layer hidden states. For forget prompts, retrieves the most
relevant forget answer via SBERT similarity, extracts forbidden token
sequences, and applies constrained beam search with:
  1. Token-level hard matching via a trie (beta threshold)
  2. SBERT semantic soft matching on last generated word (delta threshold)

Requires:
  - model_guard_target: path to the base/target LLM
  - model_guard_classifier: path to directory with classifier.pt and forget_answers.json
  - model_guard_beam_width: beam search width (default 7)
  - model_guard_beta: hard-match trie threshold (default 1.0)
  - model_guard_delta: SBERT semantic similarity threshold (default 0.5)
  - model_guard_sbert_model: SBERT model name (default "sentence-transformers/all-MiniLM-L6-v2")
  - model_guard_threshold: classifier confidence threshold (default 0.5)
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np


# ── MLP classifier (must match finetune_model_guard.py) ──────────────

class PromptClassifierMLP(nn.Module):
    def __init__(self, hidden_size, mlp_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── Trie for forbidden token sequences ───────────────────────────────

class TokenTrie:
    """Trie over tokenized forbidden phrases for prefix matching."""

    def __init__(self):
        self.root = {}

    def insert(self, token_ids):
        """Insert a sequence of token IDs into the trie."""
        node = self.root
        for tid in token_ids:
            tid = int(tid)
            if tid not in node:
                node[tid] = {}
            node = node[tid]
        node["__end__"] = True

    def check_suffix(self, generated_ids, beta=1.0):
        """Check if the suffix of generated_ids matches a forbidden sequence.

        Returns the fraction of the longest matching prefix in the trie.
        If any match ratio >= beta, the beam should be pruned.
        """
        max_match_ratio = 0.0

        for start in range(len(generated_ids)):
            node = self.root
            match_len = 0
            total_len = 0

            for i in range(start, len(generated_ids)):
                tid = int(generated_ids[i])
                if tid in node:
                    node = node[tid]
                    match_len += 1
                    # Count depth of this branch to get total phrase length
                    total_len = match_len
                    if "__end__" in node:
                        # Full match of a forbidden phrase
                        return 1.0
                else:
                    break

            if total_len > 0:
                # Estimate ratio: we matched total_len tokens into the trie
                ratio = total_len / max(total_len, 1)
                max_match_ratio = max(max_match_ratio, ratio)

        return max_match_ratio


# ── GUARD model handler ──────────────────────────────────────────────

class GUARD(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.device = model_cfg.get("device", "cuda")

        # Hyperparameters
        self.beam_width = int(model_cfg.get("model_guard_beam_width", 7))
        self.beta = float(model_cfg.get("model_guard_beta", 1.0))
        self.delta = float(model_cfg.get("model_guard_delta", 0.5))
        self.threshold = float(model_cfg.get("model_guard_threshold", 0.5))
        sbert_name = model_cfg.get("model_guard_sbert_model", "sentence-transformers/all-MiniLM-L6-v2")

        # Load tokenizer
        target_path = model_cfg.model_guard_target
        if "muse-bench" in target_path:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(target_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.special_token_ids = self.tokenizer.all_special_ids

        device_map = {"": self.device}

        # Load target LLM (frozen)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path,
            torch_dtype=torch.float16,
            device_map=device_map,
        ).eval()

        # Load SBERT for retrieval and semantic matching
        self.sbert = SentenceTransformer(sbert_name, device=self.device)

        # Load classifier and forget answers
        classifier_path = model_cfg.get("model_guard_classifier", None)
        self.classifier = None
        self.forget_answers = []
        self.forget_answer_embeddings = None
        self.tries = {}  # answer_idx -> TokenTrie

        if classifier_path:
            # Load MLP classifier
            ckpt = torch.load(
                os.path.join(classifier_path, "classifier.pt"),
                map_location=self.device, weights_only=True,
            )
            hidden_size = ckpt["hidden_size"]
            mlp_hidden = ckpt.get("mlp_hidden", 256)
            self.classifier = PromptClassifierMLP(hidden_size, mlp_hidden).to(self.device)
            self.classifier.load_state_dict(ckpt["state_dict"])
            self.classifier.eval()
            print(f"Loaded GUARD classifier (eval_acc={ckpt.get('best_eval_acc', 'N/A')})")

            # Load forget answers
            answers_path = os.path.join(classifier_path, "forget_answers.json")
            if os.path.exists(answers_path):
                with open(answers_path, 'r') as f:
                    self.forget_answers = json.load(f)
                print(f"Loaded {len(self.forget_answers)} forget answers")

                # Precompute SBERT embeddings for forget answers
                self.forget_answer_embeddings = self.sbert.encode(
                    self.forget_answers, convert_to_tensor=True, device=self.device,
                )

                # Build tries for each forget answer
                self._build_tries()

    def _build_tries(self):
        """Build token tries for all forget answers."""
        for idx, answer in enumerate(self.forget_answers):
            trie = TokenTrie()
            # Extract words/phrases from the answer as forbidden spans
            words = answer.split()
            # Use individual words and multi-word spans (up to 5-grams)
            for n in range(1, min(6, len(words) + 1)):
                for start in range(len(words) - n + 1):
                    phrase = ' '.join(words[start:start + n])
                    token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
                    if len(token_ids) > 0:
                        trie.insert(token_ids)
            self.tries[idx] = trie

    def _get_prompt_embedding(self, input_ids):
        """Get mean-pooled penultimate-layer hidden states for prompt classification."""
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            # Penultimate layer
            hidden = outputs.hidden_states[-2]  # (batch, seq, hidden)
            # Mean pool
            mask = (input_ids != self.tokenizer.pad_token_id).unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def _classify_prompts(self, input_ids):
        """Return list of 0/1 indicating whether each prompt is a 'forget' prompt."""
        if self.classifier is None:
            return [1] * input_ids.shape[0]

        embeddings = self._get_prompt_embedding(input_ids)
        with torch.no_grad():
            logits = self.classifier(embeddings)
            probs = torch.softmax(logits, dim=-1)
            labels = (probs[:, 1] > self.threshold).long().tolist()
        return labels

    def _retrieve_forbidden(self, text):
        """Retrieve the most relevant forget answer and its trie for a given prompt."""
        if not self.forget_answers or self.forget_answer_embeddings is None:
            return None, None, []

        prompt_embedding = self.sbert.encode([text], convert_to_tensor=True, device=self.device)
        similarities = torch.nn.functional.cosine_similarity(
            prompt_embedding, self.forget_answer_embeddings,
        )
        best_idx = similarities.argmax().item()
        best_answer = self.forget_answers[best_idx]
        best_trie = self.tries.get(best_idx, TokenTrie())

        # Extract forbidden words for SBERT semantic matching
        forbidden_words = list(set(best_answer.split()))

        return best_answer, best_trie, forbidden_words

    def _sbert_penalty(self, token_text, forbidden_words, forbidden_embeddings):
        """Compute SBERT semantic penalty for a generated word."""
        if not token_text.strip() or not forbidden_words:
            return 0.0

        token_embedding = self.sbert.encode([token_text.strip()], convert_to_tensor=True, device=self.device)
        similarities = torch.nn.functional.cosine_similarity(
            token_embedding, forbidden_embeddings,
        )
        max_sim = similarities.max().item()

        if max_sim >= self.delta:
            return float('inf')
        else:
            return max_sim

    def generate(self, input_ids, attention_mask=None, pad_token_id=None, **generation_args):
        max_new_tokens = generation_args.get('max_new_tokens', 200)
        batch_size = input_ids.shape[0]
        device = self.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Classify prompts
        prompt_labels = self._classify_prompts(input_ids)

        # For each sample in batch, retrieve forbidden info if classified as forget
        prompt_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        forbidden_info = []
        for i in range(batch_size):
            if prompt_labels[i] == 1:
                answer, trie, forbidden_words = self._retrieve_forbidden(prompt_texts[i])
                if forbidden_words:
                    forbidden_embeddings = self.sbert.encode(
                        forbidden_words, convert_to_tensor=True, device=device,
                    )
                else:
                    forbidden_embeddings = None
                forbidden_info.append((trie, forbidden_words, forbidden_embeddings))
            else:
                forbidden_info.append(None)

        # Generate one sample at a time (beam search doesn't batch well)
        all_generated = []
        for i in range(batch_size):
            sample_input = input_ids[i:i+1]
            sample_mask = attention_mask[i:i+1]

            if forbidden_info[i] is None:
                # Normal greedy/sampling decode for non-forget prompts
                generated = self._greedy_decode(
                    sample_input, sample_mask, max_new_tokens, pad_token_id, generation_args,
                )
            else:
                # Constrained beam search for forget prompts
                trie, forbidden_words, forbidden_embeddings = forbidden_info[i]
                generated = self._beam_search(
                    sample_input, sample_mask, max_new_tokens, pad_token_id,
                    trie, forbidden_words, forbidden_embeddings,
                )
            all_generated.append(generated)

        # Pad to same length
        max_len = max(g.shape[1] for g in all_generated)
        padded = []
        for g in all_generated:
            if g.shape[1] < max_len:
                pad = torch.full(
                    (1, max_len - g.shape[1]),
                    pad_token_id if pad_token_id is not None else 0,
                    device=device,
                )
                g = torch.cat([g, pad], dim=1)
            padded.append(g)

        return torch.cat(padded, dim=0)

    def _greedy_decode(self, input_ids, attention_mask, max_new_tokens, pad_token_id, generation_args):
        """Standard greedy/sampling decode for non-forget prompts."""
        device = self.device
        past_kv = None
        generated = input_ids.clone()
        current_input = input_ids
        current_mask = attention_mask.clone()

        for step in range(max_new_tokens):
            with torch.no_grad():
                out = self.target_model(
                    current_input,
                    attention_mask=current_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            logits = out.logits[:, -1, :]
            past_kv = out.past_key_values

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
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            current_mask = torch.cat([current_mask, torch.ones(1, 1, device=device)], dim=1)
            current_input = next_token

            if next_token.item() in self.special_token_ids:
                break

        return generated

    def _beam_search(self, input_ids, attention_mask, max_new_tokens, pad_token_id,
                     trie, forbidden_words, forbidden_embeddings):
        """Constrained beam search with trie + SBERT penalties."""
        device = self.device
        beam_width = self.beam_width
        prompt_len = input_ids.shape[1]

        # Initialize beams: (token_ids, score, past_kv, attention_mask)
        # First, run the prompt through the model to get initial past_kv
        with torch.no_grad():
            out = self.target_model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        initial_logits = out.logits[:, -1, :]
        initial_past = out.past_key_values

        # Get top beam_width*2 candidates for first step
        log_probs = torch.log_softmax(initial_logits, dim=-1)
        top_scores, top_indices = torch.topk(log_probs[0], beam_width * 2)

        beams = []
        for j in range(min(beam_width * 2, top_indices.shape[0])):
            token_id = top_indices[j].item()
            score = top_scores[j].item()

            # Check trie penalty
            generated_suffix = [token_id]
            match_ratio = trie.check_suffix(generated_suffix, self.beta)
            if match_ratio >= self.beta:
                continue

            # Check SBERT penalty
            token_text = self.tokenizer.decode([token_id])
            sbert_pen = 0.0
            if forbidden_embeddings is not None:
                sbert_pen = self._sbert_penalty(token_text, forbidden_words, forbidden_embeddings)
                if sbert_pen == float('inf'):
                    continue

            beam_score = score - sbert_pen
            new_ids = torch.cat([input_ids[0], torch.tensor([token_id], device=device)])
            beams.append((new_ids, beam_score, initial_past, token_id in self.special_token_ids))

        # Keep top beam_width
        beams.sort(key=lambda x: x[1], reverse=True)
        beams = beams[:beam_width]

        if not beams:
            # All beams pruned - fall back to greedy
            return torch.cat([input_ids, torch.argmax(initial_logits, dim=-1, keepdim=True)], dim=1)

        for step in range(1, max_new_tokens):
            if all(b[3] for b in beams):  # all finished
                break

            candidates = []
            for beam_ids, beam_score, beam_past, beam_finished in beams:
                if beam_finished:
                    candidates.append((beam_ids, beam_score, beam_past, True))
                    continue

                # Run model on last token
                last_token = beam_ids[-1:].unsqueeze(0).to(device)
                beam_mask = torch.ones(1, beam_ids.shape[0], device=device)

                with torch.no_grad():
                    out = self.target_model(
                        last_token,
                        attention_mask=beam_mask,
                        past_key_values=beam_past,
                        use_cache=True,
                    )

                logits = out.logits[:, -1, :]
                new_past = out.past_key_values
                log_probs = torch.log_softmax(logits, dim=-1)

                # Get top candidates
                top_scores, top_indices = torch.topk(log_probs[0], beam_width * 2)

                for j in range(top_indices.shape[0]):
                    token_id = top_indices[j].item()
                    score = beam_score + top_scores[j].item()

                    # Check trie penalty on generated suffix (after prompt)
                    generated_suffix = beam_ids[prompt_len:].tolist() + [token_id]
                    match_ratio = trie.check_suffix(generated_suffix, self.beta)
                    if match_ratio >= self.beta:
                        continue

                    # Check SBERT penalty
                    token_text = self.tokenizer.decode([token_id])
                    sbert_pen = 0.0
                    if forbidden_embeddings is not None:
                        sbert_pen = self._sbert_penalty(token_text, forbidden_words, forbidden_embeddings)
                        if sbert_pen == float('inf'):
                            continue

                    new_score = score - sbert_pen
                    new_ids = torch.cat([beam_ids, torch.tensor([token_id], device=device)])
                    is_finished = token_id in self.special_token_ids
                    candidates.append((new_ids, new_score, new_past, is_finished))

            if not candidates:
                break

            # Keep top beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        # Return best beam
        best_ids = beams[0][0].unsqueeze(0)
        return best_ids

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        """Forward pass - just runs the target model (GUARD only modifies generation)."""
        device = self.device

        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)

        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, device=device)

        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs,
            )

        # Compute loss if labels provided
        if labels is not None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            outputs.loss = loss

        return outputs
