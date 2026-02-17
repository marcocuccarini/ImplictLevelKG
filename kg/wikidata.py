import json
from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataClient:
    def __init__(self, cache_file):
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.cache_file = cache_file
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        except:
            self.cache = {}

    def save(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def resolve_entity(self, label):
        if label in self.cache:
            return self.cache[label].get("qid")

        q = f'SELECT ?item WHERE {{ ?item rdfs:label "{label}"@en }} LIMIT 1'
        self.sparql.setQuery(q)
        self.sparql.setReturnFormat(JSON)

        try:
            r = self.sparql.query().convert()
            if not r["results"]["bindings"]:
                return None
            qid = r["results"]["bindings"][0]["item"]["value"]
            self.cache[label] = {"qid": qid, "triples": []}
            return qid
        except:
            return None

    def extract_triples(self, qid, label):
        if self.cache.get(label, {}).get("triples"):
            return self.cache[label]["triples"]

        q = f"""
        SELECT ?pLabel ?vLabel WHERE {{
            <{qid}> ?p ?v .
            FILTER(isIRI(?v))
            ?prop wikibase:directClaim ?p .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 100
        """
        self.sparql.setQuery(q)
        self.sparql.setReturnFormat(JSON)

        try:
            r = self.sparql.query().convert()
            triples = [[label.lower(), b["pLabel"]["value"].lower(), b["vLabel"]["value"].lower()]
                       for b in r["results"]["bindings"]]
            self.cache[label]["triples"] = triples
            self.save()
            return triples
        except:
            return []
