import fitz  # PyMuPDF
import re
import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# -----------------------------
# TEXT SANITIZATION
# -----------------------------
def sanitize_for_llm(text: str) -> str:
    """
    Remove control characters and problematic PDF artifacts
    that commonly break JSON responses.
    """
    cleaned = []
    for ch in text:
        code = ord(ch)
        if code in (9, 10, 13):  # keep tab/newline/carriage return
            cleaned.append(ch)
        elif code >= 32:
            cleaned.append(ch)
        else:
            cleaned.append(" ")

    cleaned = "".join(cleaned)

    # Remove escaped hex artifacts like \x02
    cleaned = re.sub(r"\\x[0-9A-Fa-f]{2}", " ", cleaned)

    # Normalize whitespace
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


# -----------------------------
# SENTENCE CHUNKING
# -----------------------------
_SENTENCE_END_RE = re.compile(r'(?<=[\.\?\!])\s+')

def chunk_text_for_llm(text: str, max_sentences: int = 6) -> list[str]:
    """
    Break text into smaller sentence groups to avoid model overload.
    No limits on returned matches — just safe chunking.
    """
    text = text.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        sentences = _SENTENCE_END_RE.split(para)
        for i in range(0, len(sentences), max_sentences):
            group = " ".join(
                s.strip() for s in sentences[i:i + max_sentences] if s.strip()
            )
            if group:
                chunks.append(group)

    if not chunks and text.strip():
        chunks = [text.strip()]

    return chunks


# -----------------------------
# JSON RECOVERY (STRICT LLM ONLY)
# -----------------------------
def extract_json_blob_from_text(raw: str):
    """
    Attempt to recover valid JSON if model outputs stray text.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    starts = [m.start() for m in re.finditer(r'[\{\[]', raw)]
    ends = [m.start() for m in re.finditer(r'[\}\]]', raw)]

    if not starts or not ends:
        raise ValueError("No JSON delimiters found")

    for s in starts:
        for e in reversed(ends):
            if e <= s:
                continue
            candidate = raw[s:e + 1]
            try:
                return json.loads(candidate)
            except Exception:
                continue

    raise ValueError("Could not parse JSON")


# -----------------------------
# LLM EXTRACTION (STRICT)
# -----------------------------
def extract_sentences_from_chunk(chunk_text: str, user_criteria: str) -> list[str]:
    sanitized = sanitize_for_llm(chunk_text)
    if not sanitized:
        return []

    system_prompt = """
You are a STRICT literal extractor.

Your task:
Return ONLY exact verbatim sentences from the SOURCE TEXT that satisfy the user's criteria.

Rules:
1) Output exactly one valid JSON object.
2) The JSON must be: {"sentences": ["exact sentence 1.", "exact sentence 2."]}
3) Each sentence must be copied exactly from the SOURCE TEXT.
4) Do NOT paraphrase, summarize, modify punctuation, or alter whitespace.
5) Do NOT include commentary, explanations, or markdown.
6) If no matches exist, return {"sentences": []}.
"""

    user_prompt = f"User Criteria:\n{user_criteria}\n\nSOURCE TEXT:\n{sanitized}"

    attempts = 3
    backoff = 1

    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            raw = response.choices[0].message.content

            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = extract_json_blob_from_text(raw)

            if not isinstance(parsed, dict) or "sentences" not in parsed:
                raise ValueError("Invalid JSON structure")

            sentences = parsed.get("sentences") or []
            return [s for s in sentences if isinstance(s, str) and s.strip()]

        except Exception as e:
            time.sleep(backoff)
            backoff *= 2
            last_error = e

    print(f"LLM extraction failed for chunk: {last_error}")
    return []


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_and_clean_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    extracted_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")

        text = raw_text.replace('\n\n', '<PARA>')
        text = text.replace('\n', ' ')
        cleaned_text = text.replace('<PARA>', '\n\n')
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text).strip()

        extracted_pages.append({
            "page_num": page_num,
            "text": cleaned_text
        })

    doc.close()
    return extracted_pages


# -----------------------------
# HIGHLIGHTING
# -----------------------------
def normalize_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def apply_highlights_to_pdf(input_pdf: str, output_pdf: str, extracted_highlights: list[dict]):
    doc = fitz.open(input_pdf)

    for item in extracted_highlights:
        page = doc[item["page_num"]]
        raw_page_text = page.get_text("text")
        normalized_page_text = normalize_spaces(raw_page_text)

        for sentence in item["sentences"]:
            rects = page.search_for(sentence)

            if not rects:
                normalized_sentence = normalize_spaces(sentence)
                if normalized_sentence in normalized_page_text:
                    snippet = normalized_sentence[:80]
                    rects = page.search_for(snippet)

            if not rects:
                words = sentence.split()
                for i in range(max(1, len(words) - 9)):
                    snippet = " ".join(words[i:i + 10])
                    rects = page.search_for(snippet)
                    if rects:
                        break

            for r in rects:
                highlight = page.add_highlight_annot(r)
                highlight.update()

    doc.save(output_pdf)
    doc.close()
    print(f"Success! Highlighted PDF saved to: {output_pdf}")



# Initialize the Groq client (Make sure GROQ_API_KEY is in your environment variables)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_sentences_from_chunk(chunk_text: str, user_criteria: str, max_retries: int = 2) -> list[str]:
    """
    Passes a text chunk to Groq and strictly extracts verbatim sentences 
    matching the user criteria using JSON mode.
    """
    
    # Simplified system prompt that is fully compatible with JSON mode.
    # Avoid asking for any rich formatting (bold, lead-ins, wrappers) since
    # the output MUST be a flat JSON object.
    system_prompt = """You are a precise text extraction agent. Extract exact, verbatim sentences from the provided text that match the user's criteria.

RULES:
1. Copy sentences EXACTLY as they appear in the source. Do NOT alter, summarize, or rephrase anything.
2. Extract full sentences only.
3. Respond ONLY with a JSON object: {"sentences": ["sentence1", "sentence2", ...]}
4. If no sentences match, respond with: {"sentences": []}
5. Do NOT include any commentary, explanation, or formatting outside the JSON object.
6. Escape any special characters properly for valid JSON."""
    
    # Sanitize source text: replace characters that commonly break JSON generation
    sanitized_text = chunk_text.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')
    
    # Cap text length to avoid the model producing too many sentences
    # and hitting the max completion tokens limit
    MAX_CHARS = 4000
    if len(sanitized_text) > MAX_CHARS:
        sanitized_text = sanitized_text[:MAX_CHARS]
    
    user_prompt = f"Criteria: {user_criteria}\n\nSource Text:\n{sanitized_text}"
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=4096
            )
            
            # Parse the JSON string returned by Groq
            result_json = json.loads(response.choices[0].message.content)
            return result_json.get("sentences", [])
            
        except Exception as e:
            if attempt < max_retries:
                print(f"  Attempt {attempt + 1} failed, retrying... ({e})")
            else:
                print(f"  Extraction failed after {max_retries + 1} attempts: {e}")
                return []

# --- PDF Processing Logic ---
if __name__ == "__main__":

    input_pdf_path = "input.pdf"
    output_pdf_path = "highlighted_output.pdf"

    if not os.path.exists(input_pdf_path):
        print("PDF file not found.")
        exit()

    print("Extracting text...")
    pages = extract_and_clean_pages(input_pdf_path)

    criteria = """
Extract exact verbatim sentences that:
1) Explicitly state main contributions or core achievements,
2) State important theoretical results (theorems, bounds, proofs),
3) State strong empirical or conclusive findings.
"""

    all_highlights = []

    for page in pages:
        print(f"Analyzing Page {page['page_num'] + 1}...")
        chunks = chunk_text_for_llm(page["text"])

        page_matches = []
        for chunk in chunks:
            matches = extract_sentences_from_chunk(chunk, criteria)
            for m in matches:
                if m not in page_matches:
                    page_matches.append(m)

        if page_matches:
            print(f"  --> Found {len(page_matches)} matches.")
            all_highlights.append({
                "page_num": page["page_num"],
                "sentences": page_matches
            })

    if all_highlights:
        print("Applying highlights...")
        apply_highlights_to_pdf(input_pdf_path, output_pdf_path, all_highlights)
    else:
        # 2. Extract Text
        print(f"Extracting text from {input_pdf_path}...")
        pages = extract_and_clean_pages(input_pdf_path)
        
        all_highlights = []
        criteria = """Extract sentences that fall into these categories:

1. Main Contributions: Sentences where the authors describe their own novel work. Look for phrases like "We propose," "We introduce," "We show," "We achieve," "Our method," or comparative claims like "outperforms" and "state-of-the-art."

2. Key Equations or Figures: Sentences that define equations, mathematical formulations, variables, or reference specific figures/tables. Look for mathematical notation, "Equation," "defined as," "we compute," or variable definitions.

3. Important Observations: Sentences with qualitative insights, hypotheses, or design justifications. Look for "We observe," "We suspect," "This suggests," "shows that," "to counteract," or similar analytical language.

IMPORTANT: Only extract sentences that are the AUTHORS' own contributions or observations. Skip sentences that merely cite or describe prior work."""
        
        # 3. Analyze per page with LLM
        for page in pages:
            print(f"Analyzing Page {page['page_num'] + 1}...")
            matches = extract_sentences_from_chunk(page['text'], criteria)
            
            if matches:
                print(f"  --> Found {len(matches)} matches.")
                all_highlights.append({
                    "page_num": page["page_num"],
                    "sentences": matches
                })
        
        # 4. Apply Highlights
        if all_highlights:
            print(f"Applying highlights to {output_pdf_path}...")
            apply_highlights_to_pdf(input_pdf_path, output_pdf_path, all_highlights)
        else:
            print("No matches found based on the criteria.")
