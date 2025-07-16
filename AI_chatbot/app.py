import streamlit as st
import os
from dotenv import load_dotenv
from io import BytesIO
import base64
import whisper

# AI/ML Libraries
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Audio Libraries 
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

# Load Environment Variables and Configure APIs 
load_dotenv()
# Configure Google Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.error("Google API Key not found. Please add it to your .env file.")


# Caches(saving in memory) the Whisper model only once, even across reruns
@st.cache_resource(show_spinner="Loading speech-to-text model....") 
def load_whisper_model(): # Loads the Whisper model and caches it
    model = whisper.load_model("base") # base is a good balance of speed and accuracy
    return model


# 1. CORE CHATBOT LOGIC (RAG - Retrieval-Augmented Generation) 
@st.cache_resource(show_spinner="Loading and processing documents...")
def setup_rag_pipeline(pdf_paths, csv_paths):
    # Sets up the RAG pipeline using Google Gemini for chat and embeddings
    all_docs = []
    
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(file_path=path)
            all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading PDF {path}: {e}")

    for path in csv_paths:
        try:
            loader = CSVLoader(file_path=path, source_column="answer")
            all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading CSV {path}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, convert_system_message_to_human=True)

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    Be friendly, conversational, and helpful. If you don't know the answer, just say that you don't have that information.

    <context>
    {context}
    </context>

    Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


# 2. AUDIO PROCESSING FUNCTIONS 
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {e}")
        return None

def speech_to_text(audio_bytes):
    # Transcribes audio to text using the local Whisper model
    model = load_whisper_model()
    try:
        # Save audio bytes to a temporary file for Whisper to process
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        
        # Transcribe the audio file using the local model
        result = model.transcribe("temp_audio.wav", fp16=False) # fp16=False is recommended for CPU-only usage
        
        # Clean up the temporary file
        os.remove("temp_audio.wav")
        return result["text"]
    except Exception as e:
        st.error(f"Error in speech-to-text conversion: {e}")
        return ""

def autoplay_audio(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio controls autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)


# 3. STREAMLIT UI 
st.set_page_config(page_title="AI Customer Assistant", layout="wide")
st.title("AI Customer Support Assistant")
st.markdown("Ask me anything about our company! You can type or use your voice.")

# Check for necessary external dependencies
if not os.path.exists('data'):
    os.makedirs('data')
    st.warning("'data' folder created. Please add your PDF and CSV files there.")

data_dir = "data"
pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]

if not pdf_files and not csv_files:
    st.error("No PDF or CSV files found in the 'data' directory. Please add your company documents.")
else:
    retrieval_chain = setup_rag_pipeline(pdf_files, csv_files)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    if "audio_processed" not in st.session_state:
        st.session_state.audio_processed = False

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    col1, col2 = st.columns([4, 1])

    with col1:
        if prompt := st.chat_input("What is your question?"):
            st.session_state.audio_processed = False
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking with Gemini..."):
                    response = retrieval_chain.invoke({"input": prompt})
                    answer = response['answer']
                    st.markdown(answer)
                    audio_bytes = text_to_speech(answer)
                    if audio_bytes:
                        autoplay_audio(audio_bytes)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    with col2:
        st.write("Or ask by voice:")
        audio_info = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key='recorder')
        
        if audio_info and audio_info['bytes'] and not st.session_state.audio_processed:
            st.session_state.audio_processed = True
            with st.spinner("Transcribing your voice..."):
                user_speech_text = speech_to_text(audio_info['bytes'])

            if user_speech_text:
                st.session_state.messages.append({"role": "user", "content": user_speech_text})
                with st.chat_message("user"):
                    st.markdown(user_speech_text)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking with Gemini..."):
                        response = retrieval_chain.invoke({"input": user_speech_text})
                        answer = response['answer']
                        st.markdown(answer)
                        audio_bytes = text_to_speech(answer)
                        if audio_bytes:
                            autoplay_audio(audio_bytes)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

if not audio_info or not audio_info['bytes']:
    st.session_state.audio_processed = False
    
    
    
    
# .\venv\Scripts\Activate.ps1       [To Activate virtual environment]
# $env:KMP_DUPLICATE_LIB_OK="TRUE"  [Set the environment variable to prevent the crash] 