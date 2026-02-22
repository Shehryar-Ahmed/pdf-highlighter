import fitz  # PyMuPDF
import re
import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_and_clean_pages(pdf_path: str) -> list[dict]:
    """
    Extracts text page-by-page and sanitizes PDF formatting artifacts.
    Returns a list of dictionaries containing page numbers and cleaned text.
    """
    doc = fitz.open(pdf_path)
    extracted_pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")
        
        # Industry Standard Sanitization:
        # 1. Replace single newlines with spaces to fix broken sentences.
        # 2. Preserve double newlines (paragraphs) using a temporary placeholder.
        text = raw_text.replace('\n\n', '<PARAGRAPH_BREAK>')
        text = text.replace('\n', ' ')
        cleaned_text = text.replace('<PARAGRAPH_BREAK>', '\n\n')
        
        # Clean up multiple spaces
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text).strip()
        
        extracted_pages.append({
            "page_num": page_num,
            "text": cleaned_text
        })
        
    doc.close()
    return extracted_pages

def apply_highlights_to_pdf(input_pdf: str, output_pdf: str, extracted_highlights: list[dict]):
    """
    Takes the exact sentences identified by the LLM and highlights them.
    Expects extracted_highlights to be format: [{"page_num": 0, "sentences": ["match 1", "match 2"]}]
    """
    doc = fitz.open(input_pdf)
    
    for item in extracted_highlights:
        page_num = item["page_num"]
        sentences_to_highlight = item["sentences"]
        
        # Load the specific page
        page = doc[page_num]
        
        for sentence in sentences_to_highlight:
            if sentence.strip() == "" or sentence == "NONE":
                continue
                
            # search_for returns a list of Rect (rectangle coordinates) for the text
            text_instances = page.search_for(sentence)
            
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
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
    
    # 1. Check if the file exists
    if not os.path.exists(input_pdf_path):
        print(f"Error: {input_pdf_path} not found. Please place a PDF in the directory.")
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