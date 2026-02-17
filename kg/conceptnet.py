# Save this as conceptnet_local.py

class ConceptNetClient:
    def __init__(self, file_path="conceptnet-assertions-5.7.0.csv"):
        self.file_path = file_path

    def get_triples(self, variants):
        triples = []
        with open(self.file_path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                start, rel, end = parts[2], parts[1], parts[3]
                for v in variants:
                    if f"/c/en/{v}" in start or f"/c/en/{v}" in end:
                        # convert to simple labels
                        start_label = start.split("/c/en/")[-1].split("/")[0]
                        end_label = end.split("/c/en/")[-1].split("/")[0]
                        rel_label = rel.split("/r/")[-1].split("/")[0]
                        triples.append([start_label.lower(), rel_label.lower(), end_label.lower()])
        return triples



