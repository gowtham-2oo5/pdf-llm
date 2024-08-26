# Chat with PDF using Gemini

This Streamlit app allows you to interact with PDF documents by asking questions in any style—whether in layman's terms or technical language. It uses LangChain and Google Generative AI (Gemini) to process the PDFs, extract relevant information, and provide detailed responses based on the content within the documents.

## Features

- **PDF Upload**: Upload multiple PDF files and extract text, including tables and sections.
- **Text Chunking**: Split extracted text into manageable chunks for efficient processing.
- **Vector Store Creation**: Create a local FAISS vector store from the text chunks for efficient similarity search.
- **Question Answering**: Ask questions in any format and receive detailed, context-aware responses using Google Generative AI (Gemini).
- **Flexible Response Style**: The model adapts its response style based on the input, handling both layman and technical language.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/gowtham-2oo5/pdf-llm.git
   cd pdf-llm
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   # For windows:
   venv\Scripts\activate
   # For macOS and Linux
   source venv/bin/activate
   ```
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your API keys. [See here](#environment-variables) for details.

## Usage

1. Run the app:

   ```bash
   streamlit run app.py
   ```

2. Upload your PDF files via the sidebar.
3. Ask questions in the input field and receive responses based on the content of your PDFs.

## Environment Variables

Create a `.env` file in the root directory with the following content:

```env
GROQ_API_KEY = YOUR_GROQ_API_KEY
GOOGLE_API_KEY = YOUR_GEMENI_API_KEY
```

## Getting API Keys

- **GROQ API KEY:** Obtain your GROQ API key from [here](https://console.groq.com/playground).
- **GOOGLE API KEY:** Obtain your GOOGLE API key from [here](https://aistudio.google.com/app/apikey).
