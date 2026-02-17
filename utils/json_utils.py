import json
import re

def safe_json_load(x):
    if not x:
        return None
    try:
        return json.loads(re.sub(r"```json|```", "", x).strip())
    except:
        return None
