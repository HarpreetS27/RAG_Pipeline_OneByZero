from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
import os
import pickle
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')

NVIDIA_API_KEY=os.environ.get('NVIDIA_API_KEY')

st.set_page_config(layout="wide")

with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:",
                                          accept_multiple_files=True)
        submitted = st.form_submit_button("Upload!")
    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())


##########################################
# make sure to export your NVIDIA AI Playground key as NVIDIA_API_KEY!
llm = ChatNVIDIA(model="mixtral_8x7b", verify=False)
document_embedder = NVIDIAEmbeddings(model="nvolvega_40k",
                                     model_type="passage",api_key=NVIDIA_API_KEY)
query_embedder = NVIDIAEmbeddings(model="nvolveqa_40k",
                                  model_type="query")



################################################
# Option to use an existing vector store
use_existing_vector_store = st.radio(
    "Use existing vector store", ["Yes", "No"])
# Path to the vector store file
vector_store_path = "vectorstore.pkl"
# Load raw documents
raw_documents = DirectoryLoader(DOCS_DIR).load()
# Check for existing vector store
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None

if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
else:
    if raw_documents:
        # Splitting documents
        text_splitter = CharacterTextSplitter(chunk_size=2000)
        documents = text_splitter.split_documents(raw_documents)

        # Creating vector store
        vectorstore = FAISS.from_documents(documents, document_embedder)

        # Saving vector store
        with open(vector_store_path, "wb") as f:
            pickle.dump(vectorstore, f)
