import math

import ollama


class OllamaChat:
    def __init__(self, model, top_logprobs=5):
        self.model = model
        self.top_logprobs = top_logprobs

    def send_prompt(self, prompt):
        """Legacy method: returns only the raw text content (no confidence info)."""
        content, _ = self.send_prompt_with_confidence(prompt)
        return content

    def send_prompt_with_confidence(self, prompt):
        """
        Sends a prompt and returns (content, entropy_confidence).

        entropy_confidence is derived from the real token-level probabilities
        returned by Ollama (logprobs), NOT from any value the model claims
        about itself. It is computed as follows, for each generated token:

          1. Ollama returns the top-k candidate tokens with their log-probabilities
             (an approximation of the full output distribution at that step).
          2. We convert those log-probabilities to probabilities and compute the
             Shannon entropy of that (partial) distribution:
                 H = -sum(p_i * log2(p_i))
          3. We normalize by the maximum possible entropy for k candidates
             (log2(k)), giving a value in [0, 1] where 0 = maximally uncertain
             and 1 = fully certain among the observed candidates.
          4. confidence = 1 - normalized_entropy

        The final confidence is the mean of the per-token confidences across
        the whole generated response ("average token entropy" confidence).

        Returns confidence = None if logprobs are unavailable (e.g. call failed
        or the installed Ollama server does not support logprobs).
        """
        try:
            r = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                logprobs=True,
                top_logprobs=self.top_logprobs,
            )
        except Exception:
            return None, None

        content = None
        try:
            content = r["message"]["content"].strip()
        except Exception:
            pass

        confidence = self._entropy_confidence(r.get("logprobs") if hasattr(r, "get") else getattr(r, "logprobs", None))

        return content, confidence

    def _entropy_confidence(self, logprobs):
        if not logprobs:
            return None

        max_entropy = math.log2(self.top_logprobs) if self.top_logprobs > 1 else 1.0
        per_token_confidences = []

        for step in logprobs:
            candidates = getattr(step, "top_logprobs", None)
            if candidates is None and isinstance(step, dict):
                candidates = step.get("top_logprobs")
            if not candidates:
                continue

            probs = []
            for c in candidates:
                lp = getattr(c, "logprob", None)
                if lp is None and isinstance(c, dict):
                    lp = c.get("logprob")
                if lp is None:
                    continue
                probs.append(math.exp(lp))

            if not probs:
                continue

            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            normalized_entropy = min(max(normalized_entropy, 0.0), 1.0)
            per_token_confidences.append(1.0 - normalized_entropy)

        if not per_token_confidences:
            return None

        return sum(per_token_confidences) / len(per_token_confidences)
