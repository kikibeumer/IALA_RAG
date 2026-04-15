#=====================================================================================================
# Same as v3, upgraded with scored + pseudo-hybrid FAISS retrieval
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
# SAFE REQUEST
# =======================

def safe_get(url):
    try:
        return requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {url} → {e}")
        return None


# =======================
# LIST GITHUB PDFs
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

        st.write(f"Found {len(pdf_urls)} PDFs")

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
# SPLIT
# =======================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=400,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


# =======================
# VECTOR DB
# =======================

@st.cache_resource
def build_vector_store():
    embeddings = init_embeddings()

    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    raw_docs = load_pdfs_from_github_repos(GITHUB_REPOS)

    if not raw_docs:
        raise ValueError("No documents loaded from GitHub")

    splits = split_documents(raw_docs)

    embedding_dim = len(embeddings.embed_query("hello world"))

    index = faiss.IndexHNSWFlat(embedding_dim, 40)
    index.hnsw.efConstruction = 400
    index.hnsw.efSearch = 50

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
# 🔥 IMPROVED RETRIEVAL + QA
# =======================

def get_answer(query, vector_store, llm):

    THRESHOLD = 0.35

    # -----------------------------
    # 1. QUERY EXPANSION (pseudo-hybrid)
    # -----------------------------
    queries = [
        query,
        query + " definition",
        query + " standard",
        query + " guideline",
        query + " recommendation"
    ]

    # -----------------------------
    # 2. MULTI-QUERY RETRIEVAL
    # -----------------------------
    all_results = []

    for q in queries:
        results = vector_store.similarity_search_with_score(q, k=10)
        all_results.extend(results)

    # -----------------------------
    # 3. SCORE NORMALIZATION + FUSION
    # -----------------------------
    score_map = {}

    for doc, score in all_results:
        similarity = 1 / (1 + score)

        key = doc.page_content

        if key not in score_map:
            score_map[key] = {"doc": doc, "score": similarity}
        else:
            score_map[key]["score"] = max(score_map[key]["score"], similarity)

    # -----------------------------
    # 4. THRESHOLD FILTER
    # -----------------------------
    filtered = [
        v for v in score_map.values()
        if v["score"] >= THRESHOLD
    ]

    # -----------------------------
    # 5. RERANK
    # -----------------------------
    filtered.sort(key=lambda x: x["score"], reverse=True)

    top_docs = [item["doc"] for item in filtered[:4]]

    # -----------------------------
    # 6. BUILD CONTEXT
    # -----------------------------
    context = "\n\n".join(
        f"[{doc.metadata['source']} - page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in top_docs
    )

    prompt = f"""
Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content


# =======================
# STREAMLIT APP
# =======================

def main():

# ==============================================================================
    st.set_page_config(
        page_title="IALA Chat",
        layout="wide"
    )

    st.markdown("""
    <style>
    :root {
        --yellow: #fedb00;
        --light-blue: #019fe3;
        --dark-blue: #00558c;

        --title-color: #ffffff;
        --subtitle-color: #ffffff;
    }

    /* ===== FULL PAGE FIX ===== */

    /* Remove default padding/margins cleanly */
    .block-container {
        padding: 0rem 1rem 0rem 2rem !important;
    }

    html, body {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Remove Streamlit header/footer */
    header, footer {
        visibility: hidden;
    }

    /* ===== Background ===== */
    .stApp {
        background: url("https://oz3cc.dk/shutterstock_142037662.jpg") no-repeat center center fixed;
        background-size: cover;
    }

    /* Bottom yellow line */
    .stApp::after {
        content: "";
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        height: 4px;
        background: var(--yellow);
        z-index: 9999;
    }

    /* Disclaimer text */
    .stApp::before {
        content: "Gulliver is an AI model and can make mistakes.";
        position: fixed;
        left: 10px;
        bottom: 6px;
        color: white;
        font-size: 12px;
        font-style: italic;
        z-index: 9999;
    }

    /* ===== Typography ===== */
    body, .stApp {
        color: var(--dark-blue);
        font-family: 'Arial Rounded MT Bold', sans-serif;
    }

    /* Fix title position */
    h1 {
        color: var(--title-color) !important;
        font-weight: 800;
        margin-top: 0 !important;
        padding-top: 10px !important;
    }

    h2, h3 {
        color: var(--subtitle-color) !important;
    }

    /* ===== Logo ===== */
    .logo-container {
        position: absolute;
        top: 20px;  
        right: 30px;
        z-index: 1000;
    }

    .logo-container img {
        height: 110px;
        border-radius: 6px;
    }
                
    .logo-container img:hover {
        transform: translateY(-1px);
        opacity: 0.9;
    }
                
    /* ===== Buttons ===== */
    .stButton button {
        background: linear-gradient(135deg, var(--light-blue), var(--dark-blue));
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        opacity: 0.9;
    }

    /* ===== Chat ===== */

    /* Remove outer black boxes */
    [data-testid="stChatMessage"] {
        background: var(--dark-blue) !important;
        border: none !important;
        box-shadow: none !important;
        padding: 6px 0 !important;
    }

    /* Chat bubbles */
    [data-testid="stChatMessageContent"] {
        background-color: #cadde6 !important;
        color: black !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    /* Text color */
    [data-testid="stChatMessageContent"] * {
        color: black !important;
    }

    /* Avatars */
    div[data-testid="stChatMessageAvatarUser"] {
        background-color: var(--yellow) !important;
    }

    div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: var(--dark-blue) !important;
    }

    </style>
    """, unsafe_allow_html=True)


    # ===== Logo =====
    st.markdown("""
    <div class="logo-container">
        <a href="https://www.iala.int/" target="_blank">
            <img src="https://www.iala.int/content/themes/redwire-iala/assets/images/iala/logo.png">
        </a>
    </div>
    """, unsafe_allow_html=True)


    # ===== Content =====
    st.title("Ask Jonathan")
    st.markdown("Ask any question regarding the IALA Publications")

# ====================================================================================================


    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.spinner("Loading..."):
        vector_store = build_vector_store()

    st.success("Knowledge ready!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = init_llm()
                answer = get_answer(prompt, vector_store, llm)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.sidebar.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("Delete All Knowledge"):
        if os.path.exists(FAISS_PATH):
            import shutil
            shutil.rmtree(FAISS_PATH)
        st.cache_resource.clear()
        st.rerun()


if __name__ == "__main__":
    main()