import ollama

class OllamaChat:
    def __init__(self, model):
        self.model = model

    def send_prompt(self, prompt):
        try:
            r = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            return r["message"]["content"].strip()
        except:
            return None
