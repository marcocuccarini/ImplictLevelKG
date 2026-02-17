class LLMUser:
    def __init__(self, llm):
        self.llm = llm
        
    def select_relevant_kg_triples(self, text: str, kg_triples: list[tuple]) -> list[int]:
        """
        Returns the indices (1-based) of KG triples that are relevant for explaining implicit hate in the text.
        """
        prompt = PromptBuilder.select_relevant_kg_triples_prompt(text, kg_triples)
        try:
            response = self.llm.send_prompt(prompt)
            raw = getattr(response, "raw_text", "") if response else ""
            raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()

            try:
                indices = json.loads(raw)
                if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                    return indices
            except Exception as e:
                print(f"⚠️ JSON parse failed ({e}). Raw snippet: {raw[:300]!r}")

            return []
        except Exception as e:
            print(f"❌ Exception in select_relevant_kg_triples: {e}")
            return []



    def detect_and_explain_implicit_hate(self, text: str, kg_triples: list[tuple] = None) -> dict:
        """
        Takes a sentence and optional KG triples that may provide context.
        Returns a dict with:
        - implicit_hate: True/False
        - explicit_meaning: Analytical description of implied stereotype
        - confidence: Float between 0 and 1
        """
        prompt = PromptBuilder.detect_implicit_hate_prompt(text, kg_triples)
        try:
            t0 = time.time()
            response = self.llm.send_prompt(prompt)
            t1 = time.time()
            duration = t1 - t0
            raw = getattr(response, "raw_text", "") if response else ""
            print(f"⏱ detect_and_explain_implicit_hate took {duration:.2f}s — response length {len(raw)}")

            # Strip markdown code fences
            raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()

            # Attempt to parse JSON
            try:
                result = json.loads(raw)
                if all(k in result for k in ("implicit_hate", "explicit_meaning", "confidence")):
                    return result
            except Exception as e:
                print(f"⚠️ JSON parse failed ({e}). Raw snippet: {raw[:300]!r}")

            # Fallback
            return {
                "implicit_hate": None,
                "explicit_meaning": raw,
                "confidence": 0.0
            }

        except Exception as e:
            print(f"❌ Exception in detect_and_explain_implicit_hate: {e}")
            return {
                "implicit_hate": None,
                "explicit_meaning": "",
                "confidence": 0.0
            }
