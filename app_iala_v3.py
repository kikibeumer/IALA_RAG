#=====================================================================================================
# Same as v2, just cleaner code
#=====================================================================================================

import os
import streamlit as st
import PyPDF2
import faiss
import requests

from io import BytesIO
from urllib.parse import urlparse

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


# =======================
# CONFIG
# =======================

os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]

GITHUB_REPOS = [
    "https://github.com/IALAPublications/Standards",
    "https://github.com/IALAPublications/Recommendations",
    "https://github.com/IALAPublications/Guidelines",
    "https://github.com/IALAPublications/Other"
]

FAISS_PATH = "faiss_index"

REQUEST_TIMEOUT = 10

# =======================
# MODELS
# =======================

@st.cache_resource
def init_llm():
    return ChatMistralAI(model="mistral-tiny", temperature=0)


@st.cache_resource
def init_embeddings():
    return MistralAIEmbeddings(model="mistral-embed")


# =======================
# SAFE REQUEST FUNCTION
# =======================

def safe_get(url):
    try:
        return requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {url} → {e}")
        return None


# =======================
# LIST OF GITHUB PDFs
# =======================

def list_github_pdfs_from_repo(repo_url):
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        return []

    owner, repo = parts[0], parts[1]
    api_base = f"https://api.github.com/repos/{owner}/{repo}/contents"

    pdf_urls = []

    def recurse_list(path=""):
        url = f"{api_base}/{path}"
        response = safe_get(url)

        if response is None or response.status_code != 200:
            st.warning(f"Could not access {repo_url}")
            st.warning(f"{repo_url} → Status: {response.status_code}") # debugging
            return

        for item in response.json():
            if item["type"] == "file" and item["name"].lower().endswith(".pdf"):
                pdf_urls.append(item["download_url"])
            elif item["type"] == "dir":
                recurse_list(item["path"])

    recurse_list()
    return pdf_urls


# =======================
# LOAD PDFs
# =======================

def load_pdfs_from_github_repos(repo_list):
    documents = []

    for repo_link in repo_list:
        pdf_urls = list_github_pdfs_from_repo(repo_link)

        st.write(f"Found {len(pdf_urls)} PDFs") #debugging

        if not pdf_urls:
            st.warning(f"No PDFs found in {repo_link}") #debugging

        for url in pdf_urls:
            try:
                response = safe_get(url)
                if response is None or response.status_code != 200:
                    continue

                reader = PyPDF2.PdfReader(BytesIO(response.content))

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": url.split("/")[-1],
                                "page": page_num + 1
                            }
                        )
                    )

            except Exception as e:
                print(f"PDF error: {url} → {e}")

    return documents


# =======================
# SPLIT DOC INTO CHUNKS
# =======================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


# =================
# VECTOR DATABASE
# =================

@st.cache_resource
def build_vector_store():
    embeddings = init_embeddings()

    # Load existing database
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Load docs
    raw_docs = load_pdfs_from_github_repos(GITHUB_REPOS)

    if not raw_docs:
        raise ValueError("No documents loaded from GitHub")

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

    vector_store.save_local(FAISS_PATH)

    return vector_store


# =======================
# Q & A
# =======================

def get_answer(query, vector_store, llm):
    retrieved_docs = vector_store.similarity_search(query, k=4)

    context = "\n\n".join(
        f"[{doc.metadata['source']} - page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = f"""
Context:
{context}

Question: {query}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content


# =======================================
# APP
# =======================================

def main():
    st.set_page_config(
        page_title="IALA Chat",
        layout="wide"
    )

    st.title("IALA Chat")
    st.markdown("Ask questions regarding the IALA documents")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load vector DB
    with st.spinner("Loading..."):
        vector_store = build_vector_store()

    st.success("Knowledge ready!")

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
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

    # Reset chat
    if st.sidebar.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    # Full reset
    if st.sidebar.button("Delete All Knowledge"):
        if os.path.exists(FAISS_PATH):
            import shutil
            shutil.rmtree(FAISS_PATH)
        st.cache_resource.clear()
        st.rerun()


if __name__ == "__main__":
    main()