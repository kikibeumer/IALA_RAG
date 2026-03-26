import os
import streamlit as st
import PyPDF2
import faiss

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


# =======================
# Configuration
# =======================

# Use Streamlit secrets or .env instead of hardcoding
os.environ["MISTRAL_API_KEY"] = st.secrets.get("MISTRAL_API_KEY", "YOUR_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
FAISS_PATH = "faiss_index"


# =======================
# Models
# =======================

@st.cache_resource
def init_llm():
    return ChatMistralAI(
        model="mistral-tiny",
        temperature=0,
        max_retries=2,
    )


@st.cache_resource
def init_embeddings():
    return MistralAIEmbeddings(
        model="mistral-embed"
    )


# =======================
# PDF Loading
# =======================

def load_pdfs_from_folder(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1
                            }
                        )
                    )

    return documents


# =======================
# Text Splitting
# =======================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


# ===================================================================
# Vector Store (Cached, so do not need to reload all docs when running again) 
# ===================================================================

@st.cache_resource
def build_vector_store():
    embeddings = init_embeddings()

    # Load existing index (FAST)
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # ❗ First-time build (SLOW but only once)
    raw_docs = load_pdfs_from_folder(PDF_FOLDER)
    splits = split_documents(raw_docs)

    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(splits)

    # Save for next runs
    vector_store.save_local(FAISS_PATH)

    return vector_store


# =======================
# QA Logic
# =======================

def get_answer(query, vector_store, llm):
    retrieved_docs = vector_store.similarity_search(query, k=4)

    context = "\n\n".join(
        f"[{doc.metadata['source']} - page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, clearly say you don't know.

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content

# =======================
# Streamlit App
# =======================

def main():
    st.set_page_config(
        page_title="Multi-PDF Chat",
        page_icon="📄",
        layout="wide"
    )

    st.title("Chat with Your PDFs (Preloaded)")
    st.markdown("Ask questions directly about your documents.")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load vector DB automatically
    with st.spinner("Loading knowledge base..."):
        vector_store = build_vector_store()

    st.success("Knowledge base ready!")

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = init_llm()
                answer = get_answer(prompt, vector_store, llm)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    # Reset chat only
    if st.sidebar.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    # Full reset (including FAISS index)
    if st.sidebar.button("Delete Knowledge Base"):
        if os.path.exists(FAISS_PATH):
            import shutil
            shutil.rmtree(FAISS_PATH)
        st.cache_resource.clear()
        st.rerun()


if __name__ == "__main__":
    main()