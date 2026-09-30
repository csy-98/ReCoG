"""R2P generation-score utilities."""

import torch

class ConfidenceCalculator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_candidate_indices(self, tokens, target_tokens, trigger_tokens=None):
        """
        Extracts candidate indices where the next token matches the given target tokens.
        """
        candidate_indices = []
        for i, token in enumerate(tokens):
            token_clean = token.lstrip("Ġ").lower()
            if token_clean in trigger_tokens:
                j = i + 1
                while j < len(tokens):
                    next_token = tokens[j].lstrip("Ġ").lower()
                    if next_token in ['":', '"', ',', 'Ġ']:  # Skip punctuation-like tokens
                        j += 2 # changed it to 2 originally 1
                        continue
                    if next_token in target_tokens:
                        candidate_indices.append(j)
                    break  # Stop after finding the first valid token
        if not candidate_indices:
            if len(tokens) >= 3 and tokens[len(tokens) - 3] in target_tokens:
                candidate_indices = [len(tokens) - 3]
        return candidate_indices

    def compute_confidence(self, outputs, candidate_indices, token_labels):
        """
        Computes the confidence scores for the given token labels.
        """
        if not candidate_indices:
            return {label + "_confidence": 0.0 for label in token_labels} | \
                   {label + "_score": 0.0 for label in token_labels} | \
                   {"margin": 0.0}

        # Collect per-occurrence probabilities and raw logits for stability.
        probs_list = []
        logits_list = []
        key = 'scores' if 'scores' in outputs else 'logits'
        for idx in candidate_indices:
            logits = outputs[key][idx][0]  # shape: (vocab_size,)
            token_ids = [self.tokenizer.convert_tokens_to_ids(label) for label in token_labels]
            token_logits = [logits[token_id].item() for token_id in token_ids]
            probs = torch.softmax(torch.tensor(token_logits), dim=-1).tolist()
            probs_list.append(probs)
            logits_list.append(token_logits)

        # Mean probability per label across occurrences.
        avg_probs = [
            sum(p[i] for p in probs_list) / len(probs_list)
            for i in range(len(token_labels))
        ]
        # Mean raw logit per label across occurrences (kept for debugging/analysis).
        avg_logits = [
            sum(l[i] for l in logits_list) / len(logits_list)
            for i in range(len(token_labels))
        ]

        # For downstream comparisons we want a non-negative, normalized score.
        # Historically callers use `*_score` (e.g., `yes_score/no_score`), so keep it as probabilities.
        avg_scores = avg_probs

        # Margin (difference between top-1 and top-2 scores).
        sorted_scores = sorted(avg_scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        return (
            {label + "_confidence": avg_probs[i] for i, label in enumerate(token_labels)}
            | {label + "_score": avg_scores[i] for i, label in enumerate(token_labels)}
            | {label + "_logit": avg_logits[i] for i, label in enumerate(token_labels)}
            | {"margin": margin}
        )

    def calculate_confidence(self, outputs, task="recognition", target_tokens=["yes", "no"]):
        """
        Calculates the confidence scores for either 'recognition' (yes/no) or 'recall' (A/B/C) tasks.
        """
        tokens = self.tokenizer.convert_ids_to_tokens(outputs["sequences"][0])
        trigger_tokens = ["answer"]
        if task == "recognition":
            candidate_indices = self.extract_candidate_indices(tokens, ["yes", "no"], trigger_tokens)
            return self.compute_confidence(outputs, candidate_indices, ["yes", "no"])

        elif task == "recall_hard":
            candidate_indices = self.extract_candidate_indices(tokens, target_tokens, trigger_tokens)
            return self.compute_confidence(outputs, candidate_indices, target_tokens)

        else:
            raise ValueError(f"Invalid task type: {task}. Choose either 'recognition' or 'recall'.")
