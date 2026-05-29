from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.modeling_outputs import CausalLMOutput
import pickle
import json
import numpy as np

# Anchor data paths to project root (src/model/dd.py -> ../../)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class OptimizedTrigramStupidBackoff():
    def __init__(self, tokenizer, vocab_size, alpha=0.4):
        self.vocab_size = vocab_size
        self.scores = {"empty": np.zeros(self.vocab_size, dtype=np.float32)}
        self.alpha = alpha
        self.tokenizer = tokenizer
        
    def fit(self, data_paths, benchmark):
        # Step 1: Count and immediately convert to scores
        counts = {"empty": np.zeros(self.vocab_size, dtype=np.int32)}
        
        print(f"Counting from {data_paths}")
        if benchmark == "TOFU":
            for data_path in data_paths:
                with open(data_path, "r") as f:
                    for line in f:
                        line = json.loads(line)
                        tokens = self.tokenizer.tokenize(line['answer'])
                        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                        
                        for i in range(len(tokens)):
                            # Unigram counts
                            counts["empty"][tokens[i]] += 1
                            
                            # Bigram counts (context length 1)
                            if i >= 1:
                                context = str(tokens[i-1])
                                if context not in counts:
                                    counts[context] = np.zeros(self.vocab_size, dtype=np.int32)
                                counts[context][tokens[i]] += 1
                            
                            # Trigram counts (context length 2)
                            if i >= 2:
                                context = f"{tokens[i-2]}_{tokens[i-1]}"
                                if context not in counts:
                                    counts[context] = np.zeros(self.vocab_size, dtype=np.int32)
                                counts[context][tokens[i]] += 1
        elif benchmark == "MUSE":
            for data_path in data_paths:
                with open(data_path, "r") as f:
                    text = f.read()
                    tokens = self.tokenizer.tokenize(text)
                    tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                    
                    for i in range(len(tokens)):
                        # Unigram counts
                        counts["empty"][tokens[i]] += 1
                        
                        # Bigram counts (context length 1)
                        if i >= 1:
                            context = str(tokens[i-1])
                            if context not in counts:
                                counts[context] = np.zeros(self.vocab_size, dtype=np.int32)
                            counts[context][tokens[i]] += 1
                        
                        # Trigram counts (context length 2)
                        if i >= 2:
                            context = f"{tokens[i-2]}_{tokens[i-1]}"
                            if context not in counts:
                                counts[context] = np.zeros(self.vocab_size, dtype=np.int32)
                            counts[context][tokens[i]] += 1
        else:
            raise ValueError("benchmark must be either 'TOFU' or 'MUSE'")
        
        # Step 2: Convert counts to scores in-place
        print("Converting counts to scores")
        for context, context_counts in counts.items():
            total_count = np.sum(context_counts)
            if total_count > 0:
                self.scores[context] = (context_counts / total_count).astype(np.float32)
            else:
                self.scores[context] = np.zeros(self.vocab_size, dtype=np.float32)
        
        # Clear counts to free memory
        del counts
        
        # Step 3: Apply stupid backoff
        print("Applying backoff")
        
        # Cache empty scores for efficiency
        empty_scores = self.scores["empty"]
        
        # Step 3.1: Stupid backoff for bigrams (length 1 contexts)
        for context in list(self.scores.keys()):
            if context == "empty":
                continue
            parts = context.split("_")
            if len(parts) == 1:
                scores = self.scores[context]
                self.scores[context] = np.where(scores > 0, scores, self.alpha * empty_scores)
        
        # Step 3.2: Stupid backoff for trigrams (length 2 contexts)
        for context in list(self.scores.keys()):
            if context == "empty":
                continue
            parts = context.split("_")
            if len(parts) == 2:
                backoff_context = parts[1]
                scores = self.scores[context]
                if backoff_context in self.scores:
                    self.scores[context] = np.where(scores > 0, scores, self.alpha * self.scores[backoff_context])
                else:
                    self.scores[context] = np.where(scores > 0, scores, self.alpha * empty_scores)
    
    def get_scores(self, context):
        """Get probability scores for next token given context"""
        if context in self.scores:
            return self.scores[context]
        
        # Handle unseen contexts with backoff
        parts = context.split("_")
        
        if len(parts) == 1:
            return self.alpha * self.scores["empty"]
        elif len(parts) == 2:
            backoff_context = parts[1]
            if backoff_context in self.scores:
                return self.alpha * self.scores[backoff_context]
            else:
                return self.alpha**2 * self.scores["empty"]
        else:
            backoff_context = parts[-1]
            if backoff_context in self.scores:
                return self.alpha * self.scores[backoff_context]
            else:
                return self.alpha**2 * self.scores["empty"]

class DD(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.device = model_cfg.get("device", "cuda")
        if "model_dd_topk" in model_cfg:
            self.style = "topk"
            self.topk = int(model_cfg.model_dd_topk)
            if "model_dd_monte_carlo" in model_cfg:
                self.monte_carlo = model_cfg.model_dd_monte_carlo == "Yes"
        elif "model_dd_alpha" in model_cfg:
            self.style = "alpha"
            self.alpha = float(model_cfg.model_dd_alpha)
            self.log_alpha = model_cfg.get("model_dd_log_alpha", "No") == "Yes"
        else:
            raise ValueError("Must specify either 'topk' or 'alpha' in model config")

        # Check if we're using NGram models (data paths)
        self.use_ngram = (model_cfg.model_dd_use_ngram == "Yes")

        # Cross-tokenizer mode: main model and verifiers use different tokenizers
        self.cross_tokenizer = model_cfg.get("model_dd_cross_tokenizer", "No") == "Yes"

        # Initialize tokenizer and detect benchmark
        if "muse-bench" in model_cfg.model_dd_big:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            vocab_size = 32000
            benchmark = "MUSE"
            self.benchmark = "MUSE"
        else:
            vocab_size = 128256
            benchmark = "TOFU"
            self.benchmark = "TOFU"
            self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_dd_big)

        # Add padding token since Llama doesn't have one by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.special_token_ids = self.tokenizer.all_special_ids

        device_map = {"": self.device}

        # Load main model
        self.main_model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_dd_big,
            torch_dtype=torch.float16,
            device_map=device_map
        ).eval()

        # Cross-tokenizer setup: build static token mapping between main and verifier vocabs
        if self.cross_tokenizer:
            self.verifier_tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_dd_verifier_tokenizer)
            # Detect verifier vocab size via the tokenizer
            self.verifier_vocab_size = self.verifier_tokenizer.vocab_size
            # Detect main model vocab size via a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.tensor([[1]], device=self.device)
                dummy_output = self.main_model(dummy_input)
                self.main_vocab_size = dummy_output.logits.shape[-1]
            self._create_token_mapping()
            self._create_vectorized_mappings()

        if self.use_ngram:
            # Load and fit NGram models from data files
            print("Loading NGram retain model...")
            self.verifier_retain = OptimizedTrigramStupidBackoff(
                vocab_size=vocab_size,
                tokenizer=self.tokenizer
            )
            self.verifier_retain.fit(model_cfg.model_dd_retain, benchmark=benchmark)

            print("Loading NGram forget model...")
            self.verifier_forget = OptimizedTrigramStupidBackoff(
                vocab_size=vocab_size,
                tokenizer=self.tokenizer
            )
            self.verifier_forget.fit(model_cfg.model_dd_forget, benchmark=benchmark)
            
        else:
            # Load transformer verifier models
            self.verifier_retain = AutoModelForCausalLM.from_pretrained(
                model_cfg.model_dd_retain,
                torch_dtype=torch.bfloat16,
                device_map=device_map
            ).eval()

            self.verifier_forget = AutoModelForCausalLM.from_pretrained(
                model_cfg.model_dd_forget,
                torch_dtype=torch.bfloat16,
                device_map=device_map
            ).eval()
            
            # Handle topk vocabulary mask for transformer models
            if self.style == "topk":
                if model_cfg.topk_vocab == "TOFU":
                    targets = [str(_PROJECT_ROOT / "data/TOFU_downloaded/forget10.jsonl"), str(_PROJECT_ROOT / "data/TOFU_downloaded/holdout10.jsonl"), str(_PROJECT_ROOT / "data/TOFU_downloaded/retain90.jsonl")]
                    # Read and concatenate all texts first
                    print("Making vocabulary from target files...")
                    all_text = ""
                    for target in targets:
                        with open(target, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = json.loads(line)
                                all_text += line['question'] + " " + line['answer'] + " "
                elif model_cfg.topk_vocab == "MUSE":
                    targets = [str(_PROJECT_ROOT / "data/news/scal/forget_4.txt"), str(_PROJECT_ROOT / "data/news/raw/retain1.txt"), str(_PROJECT_ROOT / "data/news/raw/retain2.txt"), str(_PROJECT_ROOT / "data/news/raw/holdout.txt")]
                    print("Making vocabulary from target files...")
                    all_text = ""
                    for target in targets:
                        with open(target, 'r', encoding='utf-8') as f:
                            all_text += f.read() + " "
                else:
                    raise ValueError("topk_vocab must be either 'TOFU' or 'MUSE'")

                # Tokenize once and get unique tokens
                tokens = self.tokenizer(all_text, return_tensors="pt").input_ids
                target_tokens = set(tokens.unique().tolist())

                # Create mask for tokens not in target_tokens
                self.topk_mask = torch.ones((1, vocab_size), dtype=torch.bool, device=self.device)
                for token_id in target_tokens:
                    self.topk_mask[0, token_id] = False
                print(f"Length of target vocabulary: {len(target_tokens)} tokens (out of around 120k)")

    def _create_token_mapping(self):
        """Build bidirectional token mapping between main and verifier tokenizers."""
        print("Creating cross-tokenizer mapping (main <-> verifier)...")

        # Step 1: Exact 1:1 mappings — decode each verifier token, re-encode with main tokenizer
        self.forward_token_mapping = {}  # verifier_id -> main_id
        for token_id in range(self.verifier_vocab_size):
            token_text = self.verifier_tokenizer.decode([token_id])
            main_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
            if len(main_ids) == 1:
                self.forward_token_mapping[token_id] = main_ids[0]

        print(f"  Exact 1:1 mappings: {len(self.forward_token_mapping)}")

        # Step 2: Reverse mapping (main_id -> verifier_id) from exact matches
        self.backward_token_mapping = {v: k for k, v in self.forward_token_mapping.items()}

        # Step 3: Prefix fallback for unmapped main tokens
        verifier_text_to_id = {}
        for token_id in range(self.verifier_vocab_size):
            verifier_text_to_id[self.verifier_tokenizer.decode([token_id])] = token_id

        prefix_count = 0
        for main_id in range(self.main_vocab_size):
            if main_id in self.backward_token_mapping:
                continue
            if main_id in self.special_token_ids:
                continue
            text = self.tokenizer.decode([main_id])
            current = text
            while len(current) > 0:
                if current in verifier_text_to_id:
                    self.backward_token_mapping[main_id] = verifier_text_to_id[current]
                    prefix_count += 1
                    break
                current = current[:-1]

        print(f"  Prefix fallback mappings: {prefix_count}")
        print(f"  Total mapped: {len(self.backward_token_mapping)} / {self.main_vocab_size} main tokens")

    def _create_vectorized_mappings(self):
        """Materialize the token mapping as GPU tensors for fast indexed lookup."""
        # main_to_verifier[main_id] = verifier_id (or -1 if unmapped)
        self.main_to_verifier = torch.full((self.main_vocab_size,), -1, dtype=torch.long, device=self.device)
        self.valid_mapping_mask = torch.zeros(self.main_vocab_size, dtype=torch.bool, device=self.device)

        for main_id, verifier_id in self.backward_token_mapping.items():
            self.main_to_verifier[main_id] = verifier_id
            self.valid_mapping_mask[main_id] = True

        # Mark special tokens as "keep but don't bias"
        self.special_token_mask = torch.zeros(self.main_vocab_size, dtype=torch.bool, device=self.device)
        for sid in self.special_token_ids:
            if sid < self.main_vocab_size:
                self.special_token_mask[sid] = True

        # Combined keep mask: mapped tokens + special tokens survive; rest get -inf
        self.keep_mask = self.valid_mapping_mask | self.special_token_mask

        # Pre-compute index tensors for the scatter
        self.valid_main_ids = torch.where(self.valid_mapping_mask)[0]
        self.corresponding_verifier_ids = self.main_to_verifier[self.valid_main_ids]

        print(f"  Vectorized mappings ready: {self.valid_mapping_mask.sum().item()} valid entries")

    def _encode_for_verifier(self, raw_question=None, raw_answer=None, raw_text=None):
        """Encode raw text using the verifier's native tokenizer and chat template.

        For TOFU (instruction-tuned): applies the verifier's chat template with Q/A.
        For MUSE (base model): directly tokenizes the raw text.

        Returns:
            dict with 'input_ids' and 'attention_mask' tensors on self.device.
        """
        if self.benchmark == "TOFU" and raw_question is not None:
            # Build chat messages and apply verifier's chat template
            batch_ids = []
            for q, a in zip(raw_question, raw_answer):
                chat = [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
                try:
                    ids = self.verifier_tokenizer.apply_chat_template(
                        chat, tokenize=True, add_generation_prompt=False
                    )
                except Exception:
                    # Fallback: tokenize concatenated text if chat template fails
                    ids = self.verifier_tokenizer(
                        q + " " + a, add_special_tokens=True
                    )["input_ids"]
                batch_ids.append(ids)
            # Pad to same length
            max_len = max(len(ids) for ids in batch_ids)
            pad_id = self.verifier_tokenizer.pad_token_id or 0
            padded = [ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids]
            input_ids = torch.tensor(padded, device=self.device)
            attention_mask = (input_ids != pad_id).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        elif raw_text is not None:
            # MUSE or pretraining: directly tokenize with verifier
            enc = self.verifier_tokenizer(
                raw_text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
        else:
            return None

    def _apply_cross_tokenizer_bias(self, main_logits, verifier_diff, alpha):
        """Apply verifier logit diff onto main-model logits via precomputed token mapping.

        Args:
            main_logits: [batch, vocab_main] logits from the main model
            verifier_diff: [batch, vocab_verifier] (retain - forget) in verifier space
            alpha: steering strength
        Returns:
            adjusted logits in main-model vocab space
        """
        adjusted = main_logits.clone()
        # Only apply bias to tokens that have a valid mapping; leave unmapped tokens
        # at their original main-model logits (don't mask to -inf, as that breaks
        # eval metrics like MIA that need finite probabilities across full vocab).
        adjusted[:, self.valid_main_ids] += alpha * verifier_diff[:, self.corresponding_verifier_ids]
        return adjusted

    def _apply_cross_tokenizer_topk(self, main_logits, verifier_diff, topk, monte_carlo=False):
        """Apply topk masking across tokenizer boundary.

        Maps verifier_diff into main-model vocab space, finds bottom-k
        (most negative) differences, and masks those main-model tokens.

        Args:
            main_logits: [batch, vocab_main]
            verifier_diff: [batch, vocab_verifier] (retain - forget)
            topk: number of tokens to suppress
            monte_carlo: if True, replace masked logits with kth-largest instead of -inf
        Returns:
            adjusted logits in main-model vocab space
        """
        batch_size = main_logits.shape[0]

        # Map verifier diff into main vocab space (unmapped tokens get 0 diff).
        # Cast to main dtype: main model is fp16, verifiers are bf16; torch>=2.5
        # requires matching dtypes for this in-place index assignment.
        mapped_diff = torch.zeros_like(main_logits)
        mapped_diff[:, self.valid_main_ids] = verifier_diff[:, self.corresponding_verifier_ids].to(mapped_diff.dtype)

        # Zero out diffs for tokens not in target vocabulary (if topk_mask exists)
        if hasattr(self, 'topk_mask') and self.topk_mask is not None:
            mapped_diff[:, self.topk_mask.squeeze(0)] = 0

        # Find bottom-k tokens (most forget-biased) in main vocab space
        topk_indices = torch.topk(-mapped_diff, topk, dim=-1).indices  # [batch, topk]

        if monte_carlo:
            kth_largest = torch.topk(main_logits, topk, dim=-1).values[:, -1]  # [batch]
            batch_indices = torch.arange(batch_size, device=main_logits.device).unsqueeze(1)
            main_logits[batch_indices, topk_indices] = kth_largest.unsqueeze(1)
        else:
            mask = torch.full_like(main_logits, False, dtype=torch.bool)
            batch_indices = torch.arange(batch_size, device=main_logits.device).unsqueeze(1)
            mask[batch_indices, topk_indices] = True
            main_logits[mask] = float('-inf')

        return main_logits

    def _signed_log(self, x):
        """Apply sign(x) * log(|x| + 1) transformation."""
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    def _signed_exp(self, x):
        """Inverse of _signed_log: sign(x) * (exp(|x|) - 1)."""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def _get_ngram_context(self, generated_tokens):
        """Build context string for NGram models from generated tokens"""
        if len(generated_tokens) == 0:
            return "empty"
        elif len(generated_tokens) == 1:
            return str(generated_tokens[-1].item())
        else:
            return f"{generated_tokens[-2].item()}_{generated_tokens[-1].item()}"

    def generate(self, input_ids, attention_mask=None, pad_token_id=None,
                 raw_question=None, raw_answer=None, raw_text=None, **generation_args):
        max_new_tokens = generation_args.get('max_new_tokens', 200)
        
        # Handle batching - input_ids should be [batch_size, seq_len]
        batch_size = input_ids.shape[0]
        device = self.device
        
        # Initialize attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        
        # Move inputs to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Initialize KV caches for all models
        past_key_values_large = None
        past_key_values_retain = None if not self.use_ngram else None
        past_key_values_forget = None if not self.use_ngram else None
        
        # Track which sequences are still generating (not finished)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        generated = input_ids.clone()
        current_input = input_ids  # For first step, use full sequence
        current_attention_mask = attention_mask.clone()
        
        for step in range(max_new_tokens):
            # Skip finished sequences
            if finished.all():
                break
                
            with torch.no_grad():
                # Forward pass through main model
                outputs_large = self.main_model(
                    current_input,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values_large,
                    use_cache=True
                )
                
                # Get logits for the last token
                logits = outputs_large.logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Update KV cache for main model
                past_key_values_large = outputs_large.past_key_values
                
                if self.use_ngram:
                    # NGram-based divergence decoding
                    # Process each sequence in the batch with a simple for loop
                    for batch_idx in range(batch_size):
                        if finished[batch_idx]:
                            continue
                        
                        # Get context for this sequence
                        context = self._get_ngram_context(generated[batch_idx])
                        
                        # Get scores from both NGram models
                        retain_scores = self.verifier_retain.get_scores(context)
                        forget_scores = self.verifier_forget.get_scores(context)
                        
                        # Convert to torch tensors and move to device
                        retain_scores = torch.from_numpy(retain_scores).float().to(device)
                        forget_scores = torch.from_numpy(forget_scores).float().to(device)
                        
                        if self.style == "alpha":
                            # Add weighted difference to logits
                            # NGram scores are already probabilities, no need to transform
                            logits[batch_idx] += self.alpha * (retain_scores - forget_scores)
                        elif self.style == "topk":
                            # Get bottom k tokens based on differences and mask them
                            differences = retain_scores - forget_scores
                            topk_indices = torch.topk(-differences, self.topk).indices
                            logits[batch_idx][topk_indices] = float('-inf')

                elif self.cross_tokenizer:
                    # Cross-tokenizer: re-tokenize generated text for verifiers (no KV cache)
                    # Decode to plain text (stripping main model's special tokens),
                    # then re-encode with verifier tokenizer (which adds its own BOS etc.)
                    generated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                    verifier_enc = self.verifier_tokenizer(
                        generated_text, return_tensors="pt", padding=True,
                        truncation=True, add_special_tokens=True,
                    ).to(device)

                    outputs_retain = self.verifier_retain(
                        verifier_enc.input_ids,
                        attention_mask=verifier_enc.attention_mask,
                    )
                    outputs_forget = self.verifier_forget(
                        verifier_enc.input_ids,
                        attention_mask=verifier_enc.attention_mask,
                    )

                    # Last-token logits in verifier vocab space
                    verifier_diff = outputs_retain.logits[:, -1, :] - outputs_forget.logits[:, -1, :]

                    # Map onto main-model vocab via precomputed tensor mapping
                    if self.style == "alpha":
                        logits = self._apply_cross_tokenizer_bias(logits, verifier_diff, self.alpha)
                    elif self.style == "topk":
                        logits = self._apply_cross_tokenizer_topk(
                            logits, verifier_diff, self.topk,
                            monte_carlo=getattr(self, 'monte_carlo', False)
                        )

                else:
                    # Same-tokenizer transformer-based divergence decoding
                    outputs_retain = self.verifier_retain(
                        current_input,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values_retain,
                        use_cache=True
                    )
                    outputs_forget = self.verifier_forget(
                        current_input,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values_forget,
                        use_cache=True
                    )

                    logits_retain = outputs_retain.logits[:, -1, :]
                    logits_forget = outputs_forget.logits[:, -1, :]

                    # Update KV caches for verifier models
                    past_key_values_retain = outputs_retain.past_key_values
                    past_key_values_forget = outputs_forget.past_key_values

                    # Apply divergence decoding
                    if self.style == "alpha":
                        if self.log_alpha:
                            logits = self._signed_exp(
                                self._signed_log(logits) + self.alpha * (self._signed_log(logits_retain) - self._signed_log(logits_forget))
                            )
                        else:
                            logits += self.alpha * (logits_retain - logits_forget)
                    elif self.style == "topk":
                        differences = logits_retain - logits_forget
                        if hasattr(self, 'topk_mask'):
                            differences[:, self.topk_mask.squeeze(0)] = 0

                        # Get bottom k tokens (most negative differences) and mask them
                        topk_indices = torch.topk(-differences, self.topk, dim=-1).indices
                        # Create a mask for each batch item
                        mask = torch.full_like(logits, False, dtype=torch.bool)
                        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
                        mask[batch_indices, topk_indices] = True
                        logits[mask] = float('-inf')
            
            # Sample next tokens for all sequences
            if generation_args.get('do_sample', False):
                # Apply temperature if specified
                temperature = generation_args.get('temperature', 1.0)
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top_p if specified
                top_p = generation_args.get('top_p', None)
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Don't update finished sequences
            next_tokens[finished] = pad_token_id if pad_token_id is not None else 0
            
            # Append to generated sequence
            generated = torch.cat((generated, next_tokens), dim=1)
            
            # Update attention mask (extend by 1 for all sequences)
            new_attention = torch.ones(batch_size, 1, device=device)
            new_attention[finished] = 0  # Don't attend to padding tokens
            current_attention_mask = torch.cat((current_attention_mask, new_attention), dim=1)
            
            # Check for EOS tokens
            if hasattr(self, 'special_token_ids'):
                for i, token in enumerate(next_tokens.squeeze(1)):
                    if not finished[i] and token.item() in self.special_token_ids:
                        finished[i] = True
            
            # For next iteration, only use the new tokens (KV cache handles the rest)
            current_input = next_tokens
        
        return generated
    
    def forward(self, input_ids, attention_mask=None, position_ids=None,
            past_key_values=None, inputs_embeds=None, labels=None,
            use_cache=None, output_attentions=None, output_hidden_states=None,
            return_dict=None, raw_question=None, raw_answer=None, raw_text=None,
            **kwargs):
        """
        Forward pass through DD model with divergence decoding.
        """
        device = self.device

        # Move inputs to device if needed
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
            
        # Initialize attention mask if not provided
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, device=device)
        
        # Handle past_key_values - need separate caches for each model
        past_key_values_large = None
        past_key_values_retain = None  
        past_key_values_forget = None
        
        if past_key_values is not None:
            if isinstance(past_key_values, (tuple, list)) and len(past_key_values) == 3:
                past_key_values_large, past_key_values_retain, past_key_values_forget = past_key_values
            else:
                past_key_values_large = past_key_values
        
        with torch.no_grad():
            # Forward pass through main model
            outputs_large = self.main_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values_large,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs
            )
            
            # Get logits from main model [batch_size, seq_len, vocab_size]
            logits = outputs_large.logits.clone()
            
            if self.use_ngram:
                # NGram-based divergence decoding
                batch_size, seq_len, vocab_size = logits.shape
                
                # Process each batch and sequence position with simple for loops
                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        # Build context from previous tokens
                        if seq_idx == 0:
                            context = "empty"
                        elif seq_idx == 1:
                            context = str(input_ids[batch_idx, 0].item())
                        else:
                            context = f"{input_ids[batch_idx, seq_idx-2].item()}_{input_ids[batch_idx, seq_idx-1].item()}"
                        
                        # Get scores from both NGram models
                        retain_scores = self.verifier_retain.get_scores(context)
                        forget_scores = self.verifier_forget.get_scores(context)
                        
                        # Convert to torch tensors
                        retain_scores = torch.from_numpy(retain_scores).float().to(device)
                        forget_scores = torch.from_numpy(forget_scores).float().to(device)
                        
                        if self.style == "alpha":
                            # Add weighted difference to logits
                            # NGram scores are already probabilities, no need to transform
                            logits[batch_idx, seq_idx] += self.alpha * (retain_scores - forget_scores)
                        elif self.style == "topk":
                            # Get bottom k tokens based on differences and mask them
                            differences = retain_scores - forget_scores
                            topk_indices = torch.topk(-differences, self.topk).indices
                            logits[batch_idx, seq_idx, topk_indices] = float('-inf')
                
            else:
                # Transformer-based divergence decoding

                # In cross-tokenizer mode, re-tokenize inputs for the verifiers
                if self.cross_tokenizer:
                    # Use raw text to encode natively with verifier's tokenizer/chat template
                    verifier_enc = self._encode_for_verifier(
                        raw_question=raw_question, raw_answer=raw_answer, raw_text=raw_text
                    )
                    if verifier_enc is not None:
                        verifier_input_ids = verifier_enc["input_ids"]
                        verifier_attention_mask = verifier_enc["attention_mask"]
                    else:
                        # Fallback: decode from main tokenizer (strips special tokens)
                        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                        verifier_enc = self.verifier_tokenizer(
                            texts, return_tensors="pt", padding=True,
                            truncation=True, add_special_tokens=True,
                        ).to(device)
                        verifier_input_ids = verifier_enc.input_ids
                        verifier_attention_mask = verifier_enc.attention_mask
                else:
                    verifier_input_ids = input_ids
                    verifier_attention_mask = attention_mask

                outputs_retain = self.verifier_retain(
                    input_ids=verifier_input_ids,
                    attention_mask=verifier_attention_mask,
                    position_ids=None if self.cross_tokenizer else position_ids,
                    past_key_values=past_key_values_retain,
                    inputs_embeds=None if self.cross_tokenizer else inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    **(kwargs if not self.cross_tokenizer else {})
                )

                outputs_forget = self.verifier_forget(
                    input_ids=verifier_input_ids,
                    attention_mask=verifier_attention_mask,
                    position_ids=None if self.cross_tokenizer else position_ids,
                    past_key_values=past_key_values_forget,
                    inputs_embeds=None if self.cross_tokenizer else inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    **(kwargs if not self.cross_tokenizer else {})
                )

                logits_retain = outputs_retain.logits
                logits_forget = outputs_forget.logits

                if self.cross_tokenizer:
                    # Verifier logits are [batch, verifier_seq_len, verifier_vocab].
                    # We only need the last-position diff, applied per main-model position.
                    # For the full-sequence forward pass, use the last token's diff
                    # broadcast across all sequence positions (verifier seq may differ in length).
                    verifier_diff = logits_retain[:, -1, :] - logits_forget[:, -1, :]  # [batch, verifier_vocab]
                    batch_size, seq_len, _ = logits.shape
                    if self.style == "alpha":
                        for seq_idx in range(seq_len):
                            logits[:, seq_idx, :] = self._apply_cross_tokenizer_bias(
                                logits[:, seq_idx, :], verifier_diff, self.alpha
                            )
                    elif self.style == "topk":
                        for seq_idx in range(seq_len):
                            logits[:, seq_idx, :] = self._apply_cross_tokenizer_topk(
                                logits[:, seq_idx, :], verifier_diff, self.topk,
                                monte_carlo=getattr(self, 'monte_carlo', False)
                            )
                else:
                    # Same-tokenizer: apply divergence decoding directly
                    if self.style == "alpha":
                        if self.log_alpha:
                            logits = self._signed_exp(
                                self._signed_log(logits) + self.alpha * (self._signed_log(logits_retain) - self._signed_log(logits_forget))
                            )
                        else:
                            logits += self.alpha * (logits_retain - logits_forget)

                    elif self.style == "topk":
                        differences = logits_retain - logits_forget

                        if hasattr(self, 'topk_mask') and self.topk_mask is not None:
                            differences = differences.clone()
                            differences[:, :, self.topk_mask.squeeze(0)] = 0

                        batch_size, seq_len, vocab_size = differences.shape
                        differences_flat = differences.view(-1, vocab_size)
                        logits_flat = logits.view(-1, vocab_size)

                        topk_indices = torch.topk(-differences_flat, self.topk, dim=-1).indices

                        if self.monte_carlo:

                            kth_largest_logits = torch.topk(logits_flat, self.topk, dim=-1).values[:, -1]  # kth largest (last in topk)

                            # Replace the topk worst token logits with the kth largest logit value
                            batch_seq_indices = torch.arange(batch_size * seq_len, device=device).unsqueeze(1)
                            logits_flat[batch_seq_indices, topk_indices] = kth_largest_logits.unsqueeze(1)

                        else:
                            mask = torch.full_like(logits_flat, False, dtype=torch.bool)
                            batch_seq_indices = torch.arange(batch_size * seq_len, device=device).unsqueeze(1)
                            mask[batch_seq_indices, topk_indices] = True

                            logits_flat[mask] = float('-inf')
                        logits = logits_flat.view(batch_size, seq_len, vocab_size)
        
        # Create output in the same format as the main model
        modified_outputs = outputs_large
        modified_outputs.logits = logits
        
        # If using cache, return combined past_key_values for all three models
        if use_cache and modified_outputs.past_key_values is not None:
            if self.use_ngram:
                modified_outputs.past_key_values = outputs_large.past_key_values
            else:
                combined_past_key_values = (
                    outputs_large.past_key_values,
                    outputs_retain.past_key_values, 
                    outputs_forget.past_key_values
                )
                modified_outputs.past_key_values = combined_past_key_values
        
        # Handle labels for training (compute loss on modified logits)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            modified_outputs.loss = loss
        
        return modified_outputs