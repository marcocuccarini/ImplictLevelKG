class PromptBuilder:
    @staticmethod
    def detect_implicit_hate_prompt(text: str, kg_triples: list[tuple] = None) -> str:
        """
        Prompt for detecting and explaining implicit hate, optionally using KG triples.
        """
        triples_text = ""
        if kg_triples:
            triples_text = "\nKG TRIPLES:\n" + "\n".join([f"{s} - {p} - {o}" for s, p, o in kg_triples])
        
        return f"""
You are a classifier that detects **implicit hate speech**.

INPUT:
"{text}"
{triples_text}

TASK:
1. Determine whether the text contains *implicit* hate speech toward any protected group.
   Protected groups include (but are not limited to) people targeted by race, ethnicity,
   nationality, gender, sexual orientation, disability, and religion.

2. If hate is present, produce an *explicit analytical description* of what the speaker is implying.
   - Do NOT generate new slurs or hate speech.
   - Do NOT write the hateful sentiment as a direct first-person statement.
   - You must paraphrase analytically, e.g.:
       "The speaker implies that group X is inferior/violent/unwanted."

3. Provide a confidence score between 0 and 1.

4. Return ONLY valid JSON in this format:
{{
  "implicit_hate": true/false,
  "explicit_meaning": "string",
  "confidence": float
}}
"""

    def select_relevant_kg_triples_prompt(text: str, kg_triples: list[tuple]) -> str:
        """
        Prompt for selecting KG triples relevant to understanding implicit hate in a text.
        """
        triples_text = "\n".join([f"{i+1}. {s} - {p} - {o}" for i, (s, p, o) in enumerate(kg_triples)])

        return f"""
You are an assistant that selects **relevant knowledge graph (KG) triples** to help explain implicit hate in a text.

INPUT TEXT:
"{text}"

KG TRIPLES:
{triples_text}

TASK:
1. Identify which KG triples are informative for understanding the **implicit hate** (stereotypes, biases, or implied negative meanings) in the text.
2. Only select triples that provide useful context to explain the underlying implications.
3. Do NOT create new triples or alter the input triples.
4. Return a JSON array of integers corresponding to the **indices of the selected triples** (1-based indexing as in the list above).

OUTPUT FORMAT:
[integer, integer, ...]
"""

