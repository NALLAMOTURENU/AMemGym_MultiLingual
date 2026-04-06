import re
import json
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


_model_cache = {}  # Cache loaded models


def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.loads(f.read())
    

def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_json(response):
    """
    Parse JSON content from agent responses, handling various formatting issues.

    This function extracts and parses JSON from agent responses that may contain
    additional text or be wrapped in markdown code blocks. It's designed to be
    robust against common formatting variations in LLM outputs.

    Args:
        response (str): Raw response text from the agent

    Returns:
        dict or None: Parsed JSON object, or None if parsing fails
    """
    # remove ```json``` and ``` from the response
    json_part = re.search(r'```json(.*?)```', response, re.DOTALL)
    if json_part:
        response = json_part.group(1).strip()
    return json.loads(response)


def find_best_semantic_match(value, candidates, model_name="BAAI/bge-small-zh-v1.5", threshold=0.80):
    """Find best matching candidate using semantic similarity (embeddings).

    Args:
        value (str): The value to match
        candidates (list): List of candidate values from schema
        model_name (str): SentenceTransformer model to use
        threshold (float): Minimum cosine similarity (0-1) to accept a match

    Returns:
        tuple: (best_match, similarity_score) where best_match is string or None
    """
    if not candidates:
        return None, 0.0

    # Load model once and cache it
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    model = _model_cache[model_name]

    # Embed value and all candidates
    value_embed = model.encode(value, convert_to_tensor=False)
    candidate_embeds = model.encode(candidates, convert_to_tensor=False)

    # Compute cosine similarity
    scores = cosine_similarity([value_embed], candidate_embeds)[0]

    best_idx = scores.argmax()
    best_match = candidates[best_idx]
    score = float(scores[best_idx])

    if score >= threshold:
        return best_match, score
    return None, score
