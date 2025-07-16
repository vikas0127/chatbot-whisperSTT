# Project 1: Full-Stack AI Chatbot (R&D Prototype)

This project is a fully functional, voice-enabled AI chatbot built as a proof-of-concept. The primary goal was to research and demonstrate a superior solution to the company's existing Speech-to-Text (STT) and chatbot intelligence problems.

This prototype served as a successful R&D platform to validate the effectiveness of modern AI tools before building a production-ready solution.

---

## Features

-   **Voice-Enabled Interaction:** Full voice-in and voice-out capabilities.
-   **High-Accuracy STT:** Uses a locally run **Whisper** model for fast and accurate speech-to-text.
-   **Intelligent Responses:** Implements a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions based on knowledge from custom documents (PDFs and CSVs).
-   **Interactive UI:** A user-friendly web interface built with **Streamlit**.

---

## Technology Stack

-   **Application Framework:** Streamlit
-   **Core AI Logic:** LangChain
-   **Language Model (LLM):** Google Gemini
-   **Speech-to-Text (STT):** OpenAI Whisper (running locally)
-   **Text-to-Speech (TTS):** gTTS (Google Text-to-Speech)
-   **Vector Store:** FAISS (for similarity search)
-   **Language:** Python

---

## How to Run

_Note: This is a prototype and requires specific environment setup, including FFmpeg and a `GOOGLE_API_KEY` in a `.env` file._

1.  **Create and activate a virtual environment.**
2.  **Set the `KMP_DUPLICATE_LIB_OK=TRUE` environment variable** to avoid library conflicts on some systems.
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```