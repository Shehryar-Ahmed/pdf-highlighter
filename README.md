# PDF Highlighter

A Python tool that uses LLMs (via Groq) to automatically identify and highlight key sentences in PDF documents based on specific research criteria.

## Features
- **Intelligent Extraction**: Uses LLM to find verbatim sentences matching complex criteria.
- **PyMuPDF Integration**: High-fidelity text extraction and PDF annotation.
- **Customizable Criteria**: Easily change what the script looks for (e.g., contributions, equations, observations).

## Prerequisites
- Python 3.8+
- A [Groq API Key](https://console.groq.com/)

## Installation

1. **Clone the repository** (or copy the files).
2. **Install dependencies**:
   ```bash
   pip install pymupdf groq python-dotenv
   ```
3. **Set up environment variables**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

1. Place your input PDF in the project directory and name it `input.pdf` (or update the path in `file.py`).
2. Run the script:
   ```bash
   python file.py
   ```
3. The highlighted PDF will be saved as `highlighted_output.pdf`.

## Configuration

You can customize the search criteria by modifying the `criteria` variable in the `if __name__ == "__main__":` block of `file.py`. 

Current criteria:
- **Main Contributions**: Novel findings, primary goals, core achievements.
- **Key Equations or Figures**: References to math formulas or critical figures.
- **Important Observations**: Conclusive statements or significant empirical findings.

## Dependencies
- `PyMuPDF` (fitz): For PDF manipulation.
- `groq`: For LLM inference.
- `python-dotenv`: For managing environment variables.
