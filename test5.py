import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFDirectoryLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Initialize session state for file uploads and vector DB
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = {}

if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None

if "selected_file" not in st.session_state:
    st.session_state["selected_file"] = None

@st.cache_resource(show_spinner=True)
def extract_model_names(models_info: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, ...]:
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names

def create_vector_db(file_upload) -> FAISS:
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = NVIDIAEmbeddings()
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

def process_question(question: str, vector_db: FAISS, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = ChatNVIDIA(model=selected_model)
    
    retriever = vector_db.as_retriever()

    template = """ 
    Answer the question based on the provided context only
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """

    prompt = ChatPromptTemplate.from_template(template)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': question})
    logger.info("Question processed and response generated")
    return response['answer']

@st.cache_data
def extract_all_pages_as_images(file_path) -> List[Any]:
    logger.info(f"Extracting all pages as images from file: {file_path}")
    pdf_pages = []
    with pdfplumber.open(file_path) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def delete_vector_db(vector_db: Optional[FAISS]) -> None:
    logger.info("Deleting vector DB")
    if vector_db is not None:
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.session_state.pop("uploaded_files", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def main() -> None:
    st.subheader("🧠 OneByZero RAG using NVIDIA", divider="gray", anchor=False)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=False
    )

    if file_upload:
        if file_upload.name not in st.session_state["uploaded_files"]:
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, file_upload.name)
            with open(file_path, "wb") as f:
                f.write(file_upload.getvalue())
            
            st.session_state["uploaded_files"][file_upload.name] = file_path
            if st.session_state["vector_db"] is None:
                st.session_state["vector_db"] = create_vector_db(file_upload)
            st.session_state["pdf_pages"] = extract_all_pages_as_images(file_path)
            st.session_state["selected_file"] = file_upload.name

    delete_collection = col1.button("⚠️ Delete collection", type="secondary")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    selected_file = col2.selectbox(
        "Select a document to view:",
        list(st.session_state["uploaded_files"].keys()),
        key="selected_file"
    )

    if selected_file and st.session_state["selected_file"] != selected_file:
        file_path = st.session_state["uploaded_files"][selected_file]
        st.session_state["pdf_pages"] = extract_all_pages_as_images(file_path)
        st.session_state["selected_file"] = selected_file

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], "meta/llama3-70b-instruct"
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")

    if st.session_state["selected_file"]:
        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )

        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

if __name__ == "__main__":
    main()
