import streamlit as st
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Q&A(beta)",
    page_icon="📄",
    layout="wide"
)


# --- Core Application Logic ---

@st.cache_resource(show_spinner="Initializing AI models...")
def get_models(_api_key):
    """
    Initializes and caches the Gemini LLM and embeddings model.
    """
    try:
        os.environ['GOOGLE_API_KEY'] = _api_key
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing Google models: {e}", icon="🚨")
        return None, None


@st.cache_resource(show_spinner="Processing and embedding documents...")
def create_vector_store(_uploaded_files, _embeddings):
    """
    Loads, splits, and creates a FAISS vector store from the uploaded documents.
    The result is cached as a resource to avoid reprocessing the same files.
    """
    if not _uploaded_files or _embeddings is None:
        return None

    all_texts = []
    for uploaded_file in _uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            # Determine the loader based on the file extension
            _, extension = os.path.splitext(file_path)
            if extension.lower() == ".pdf":
                loader = PyPDFLoader(file_path)
            elif extension.lower() == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            elif extension.lower() == ".json":
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema='.. | .text? | .content? | strings',
                    text_content=False
                )
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}", icon="🚨")
                continue  # Skip to the next file

            # Load and split the document
            documents = loader.load()
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                all_texts.extend(texts)
            else:
                st.warning(f"Could not load any text from {uploaded_file.name}.", icon="⚠️")

        except Exception as e:
            st.error(f"An error occurred processing {uploaded_file.name}: {e}", icon="🚨")
        finally:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)

    if not all_texts:
        st.error("No text could be extracted from the uploaded documents.", icon="🚨")
        return None

    # Create the FAISS vector store from all extracted texts
    vector_store = FAISS.from_documents(all_texts, _embeddings)
    return vector_store


def get_qa_chain(llm, vector_store):
    """Creates and returns the Question-Answering chain."""
    if llm is None or vector_store is None:
        return None

    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide a detailed and descriptive answer based on the provided context.

    Context: {context}
    Question: {question}
    Helpful Answer:
    """
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain


# --- Streamlit UI ---

st.title("📄 RAG Question answering system")
st.markdown("Upload one or more documents, and I'll answer your questions about them using Google's Gemini model.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Get your API key from Google AI Studio."
    )

    st.header("📁 Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, TXT, JSON)",
        type=["pdf", "txt", "json"],
        accept_multiple_files=True
    )

    if st.button("Clear Cache & Reset"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

# --- Main Application Logic ---

# Initialize session state for the QA chain
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if not google_api_key:
    st.warning("Please enter your Google API key in the sidebar to begin.", icon="⚠️")
elif uploaded_files:
    llm, embeddings = get_models(google_api_key)
    vector_store = create_vector_store(uploaded_files, embeddings)

    if vector_store:
        st.session_state.qa_chain = get_qa_chain(llm, vector_store)
        st.success("Documents processed successfully! You can now ask questions.", icon="✅")
    else:
        st.session_state.qa_chain = None  # Ensure chain is cleared if processing fails

# --- Q&A Interaction ---
if st.session_state.qa_chain:
    st.header("❓ Ask a Question")
    query = st.text_input("Enter your question about the documents:", key="query_input")

    if query:
        with st.spinner("Searching for the answer..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": query})
                st.subheader("Answer:")
                st.write(result["result"])

                with st.expander("Show Source Chunks"):
                    st.write(result["source_documents"])
            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")
else:
    st.info("Please provide your API key and upload your documents to get started.")