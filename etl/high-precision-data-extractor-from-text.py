"""
Hybrid High-Precision Extractor

Requirements:
 - install before running):
    pip install transformers sentence-transformers dateparser numpy
 - create your Hugging Face account (if you don't have one) and authenticate (https://huggingface.co/docs/huggingface_hub/v0.27.0.rc0/en/package_reference/authentication)

What this script does:
    - Uses a CoNLL03 NER model (dbmdz/bert-large-cased-finetuned-conll03-english)
      to extract PERSON names.
    - Uses a sentence-transformers embedding model (all-MiniLM-L6-v2) to
      classify candidate numeric tokens/phrases as PHONE or SSN based on
      semantic similarity to prototypes (no regex used).
    - Uses dateparser.search.search_dates to find likely intake dates.
    - Formats phone as (XXX) XXX-XXXX and SSN as XXX-XX-XXXX or masked variants.
    - Returns a JSON string with fields: fullName, ssn, phoneNumber, intakeDate.
      If a field is not found, its value is null.

Note: This implementation avoids regex entirely and relies on token heuristics +
semantic similarity for disambiguation.
"""

import json
from typing import Optional, List, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
import dateparser
from dateparser.search import search_dates

# ----------------------------
# Model loading (NER & Embeddings)
# ----------------------------
NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load NER pipeline (Hugging Face). Aggregation groups subword tokens.
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load embedding model
embed_model = SentenceTransformer(EMBED_MODEL)

# ----------------------------
# Prototype phrases for embedding classification
# ----------------------------
PHONE_PROTOTYPES = [
    "phone number",
    "call me at",
    "contact number",
    "mobile",
    "cell phone",
    "my phone is",
    "tel",
    "telephone"
]

SSN_PROTOTYPES = [
    "social security number",
    "ssn",
    "social-security",
    "social security no",
    "ssn is",
    "my ssn is"
]

# Compute prototype embeddings
PHONE_PROT_EMBS = embed_model.encode(PHONE_PROTOTYPES, convert_to_numpy=True)
SSN_PROT_EMBS = embed_model.encode(SSN_PROTOTYPES, convert_to_numpy=True)

# ----------------------------
# Helpers (no regex)
# ----------------------------
def normalize_whitespace(s: str) -> str:
    return " ".join(s.replace("\n", " ").split())

def digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def avg_prototype_similarity(candidate_text: str, prototype_embs: np.ndarray) -> float:
    emb = embed_model.encode([candidate_text], convert_to_numpy=True)[0]
    sims = [cosine_sim(emb, p) for p in prototype_embs]
    return float(np.mean(sims))

def format_phone(raw: str) -> Optional[str]:
    digits = digits_only(raw)
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
    # handle country-code like +1XXXXXXXXXX -> take last 10 digits
    if len(digits) > 10 and digits.endswith(digits[-10:]):
        d = digits[-10:]
        return f"({d[0:3]}) {d[3:6]}-{d[6:10]}"
    return None

def format_ssn(raw: str) -> Optional[str]:
    # Full SSN (9 digits)
    digits = digits_only(raw)
    if len(digits) == 9:
        return f"{digits[0:3]}-{digits[3:5]}-{digits[5:9]}"
    # Masked forms like ***-77-9876 -> digits would be 779876 (6 digits)
    # We'll map to XXX-77-9876 if we have 6 digits and original text contains '***' or 'xxx'
    low = raw.lower()
    if ("***" in raw or "xxx" in low) and len(digits) >= 6:
        # try to split 6 digits into 2 and 4
        if len(digits) == 6:
            return f"XXX-{digits[0:2]}-{digits[2:6]}"
        # if more digits present, take the rightmost 6 as fallback
        return f"XXX-{digits[-6:-4]}-{digits[-4:]}"
    return None

# ----------------------------
# Candidate extraction heuristics (no regex)
# ----------------------------
def candidate_numeric_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Generate candidate spans that contain digits or masking characters.
    Returns list of tuples: (span_text, start_index, end_index_in_tokens)
    Approach (no regex): tokenize by whitespace and build n-grams up to length 4
    that contain at least one digit or '*' or 'x' character.
    """
    tokens = text.replace("\n", " ").split()
    candidates = []
    n = len(tokens)
    max_ngram = 4
    for i in range(n):
        for l in range(1, max_ngram + 1):
            j = i + l
            if j > n:
                break
            span = " ".join(tokens[i:j])
            # check for digits or masking characters (no regex)
            if any(ch.isdigit() for ch in span) or "*" in span or "x" in span.lower():
                candidates.append((span, i, j))
    # deduplicate while preserving order
    seen = set()
    dedup = []
    for s, a, b in candidates:
        key = (s.strip(), a, b)
        if key not in seen:
            seen.add(key)
            dedup.append((s.strip(), a, b))
    return dedup

# ----------------------------
# Main extractor
# ----------------------------
def extract_json(text: str) -> str:
    """
    Extract fullName, ssn, phoneNumber, intakeDate from free-form text.
    Returns JSON string with keys: fullName, ssn, phoneNumber, intakeDate (YYYY-MM-DD or null).
    """
    cleaned = normalize_whitespace(text)

    # 1) NER for PERSON (dbmdz CoNLL03)
    fullName = None
    try:
        ner_entities = ner_pipeline(cleaned)
        # pick first PERSON (PER) entity if available
        for ent in ner_entities:
            # dbmdz uses 'PER' or 'I-PER' depending on pipeline; pipeline returns entity_group aggregated
            if ent.get("entity_group") == "PER":
                # 'word' contains the entity string
                candidate_name = ent.get("word", "").strip()
                if candidate_name:
                    fullName = candidate_name
                    break
    except Exception:
        # fallback: try a coarse heuristic: look for honorifics + next tokens
        honorifics = {"mr", "mrs", "ms", "dr", "miss", "mx"}
        tokens = cleaned.split()
        for idx, tk in enumerate(tokens[:-1]):
            if tk.strip().lower().rstrip(".") in honorifics:
                # take next 2 tokens as name
                first = tokens[idx + 1] if idx + 1 < len(tokens) else ""
                second = tokens[idx + 2] if idx + 2 < len(tokens) else ""
                cand = f"{first} {second}".strip()
                if cand:
                    fullName = cand
                    break

    # 2) Dates: use dateparser.search.search_dates to find dates in text
    intakeDate = None
    try:
        found = search_dates(cleaned, settings={'STRICT_PARSING': False})
        # search_dates returns list of (matched_text, datetime)
        if found:
            # Heuristic: prefer earliest DATE that appears near words like "provided", "intake", "on", "dated"
            prioritized = None
            priority_keywords = ["intake", "provided", "provided info", "date", "on", "arrived", "submitted"]
            # try to find a matched_text that contains a priority keyword (no regex)
            for match_text, dt in found:
                low = match_text.lower()
                if any(k in cleaned.lower() and low in cleaned.lower() for k in priority_keywords):
                    prioritized = dt
                    break
            # fallback to first found date
            selected = prioritized if prioritized else found[0][1]
            intakeDate = selected.strftime("%Y-%m-%d")
    except Exception:
        intakeDate = None

    # 3) Numeric candidates (phones/ssn) via embedding similarity
    phoneNumber = None
    ssn = None
    candidates = candidate_numeric_spans(cleaned)

    # For ranking candidate likelihoods, compute similarity to prototypes
    # We will pick best candidate for phone and best for SSN separately
    best_phone_score = 0.0
    best_phone_text = None
    best_ssn_score = 0.0
    best_ssn_text = None

    # Embedding cache for candidate texts
    cand_texts = [c[0] for c in candidates]
    if cand_texts:
        cand_embs = embed_model.encode(cand_texts, convert_to_numpy=True)
    else:
        cand_embs = []

    for idx, (span, _, _) in enumerate(candidates):
        emb = cand_embs[idx]
        # compute avg similarity to prototypes
        # phone similarity
        phone_sims = [cosine_sim(emb, p) for p in PHONE_PROT_EMBS]
        ssn_sims = [cosine_sim(emb, p) for p in SSN_PROT_EMBS]
        phone_score = float(np.mean(phone_sims))
        ssn_score = float(np.mean(ssn_sims))

        # Heuristic boost: presence of 10+ digits favors phone; presence of 9 digits or masked '***' favors ssn
        d_only = digits_only(span)
        if len(d_only) >= 10:
            phone_score += 0.15  # boost
        if len(d_only) == 9:
            ssn_score += 0.15
        if ("***" in span) or ("xxx" in span.lower()):
            ssn_score += 0.20

        # Update bests
        if phone_score > best_phone_score and phone_score > ssn_score:
            best_phone_score = phone_score
            best_phone_text = span
        if ssn_score > best_ssn_score and ssn_score >= phone_score:
            best_ssn_score = ssn_score
            best_ssn_text = span

    # Format selected candidates
    if best_phone_text:
        formatted = format_phone(best_phone_text)
        if formatted:
            phoneNumber = formatted
    if best_ssn_text:
        formatted = format_ssn(best_ssn_text)
        if formatted:
            ssn = formatted

    # Additional fallback heuristics (in case embeddings missed):
    # - Scan tokens for any 10-digit token to format as phone
    if not phoneNumber:
        for tok in cleaned.split():
            if len(digits_only(tok)) >= 10:
                f = format_phone(tok)
                if f:
                    phoneNumber = f
                    break

    # - Scan for masked SSN with '***' presence
    if not ssn:
        for tok in cleaned.split():
            if "***" in tok or "xxx" in tok.lower():
                f = format_ssn(tok)
                if f:
                    ssn = f
                    break

    # Build result, ensuring nulls where missing
    result = {
        "fullName": fullName if fullName else None,
        "ssn": ssn if ssn else None,
        "phoneNumber": phoneNumber if phoneNumber else None,
        "intakeDate": intakeDate if intakeDate else None
    }

    return json.dumps(result, indent=4)

# ----------------------------
# Example usage (for testing)
# ----------------------------
if __name__ == "__main__":
    sample = """
    customer_notes.txt The client provided info on 09/23/2024. She goes by Mrs. Eleanor Vance, lives in the suburbs.
    Her contact number is 555-234-7890 (cell) and her social security number is ***-77-9876. We need to confirm the billing address later.
    """

    print(extract_json(sample))

