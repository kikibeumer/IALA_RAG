#=====================================================================================================
# RAG APP — FIXED VERSION
# ✅ Auto-scroll working
# ✅ Clean math rendering
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
    except:
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

        for url in pdf_urls:
            try:
                response = safe_get(url)
                if response is None or response.status_code != 200:
                    continue

                reader = PyPDF2.PdfReader(BytesIO(response.content)) # RESEARCH pdfplumber IS BETTER FOR BETTER RETRIEVAL?

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

            except:
                pass

    return documents


# =======================
# SPLIT
# =======================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=160,
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

    splits = split_documents(raw_docs)

    embedding_dim = len(embeddings.embed_query("hello"))

    index = faiss.IndexHNSWFlat(embedding_dim, 40)

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
# Q&A
# =======================

def get_answer(query, vector_store, llm):

    # 🔹 Get docs WITH scores
    results = vector_store.similarity_search_with_score(
    query,
    k=7,
    fetch_k=12
    )

    docs = [doc for doc, score in results]
    scores = [score for doc, score in results]

    # 🔹 Convert FAISS distance → similarity-like score
    # (FAISS returns distance, lower = better → invert it)
    if scores:
        max_score = max(scores)
        min_score = min(scores)

        # normalize to 0–1 (safe)
        norm_scores = [
            1 - (s - min_score) / (max_score - min_score + 1e-8)
            for s in scores
        ]

        confidence = sum(norm_scores) / len(norm_scores)
    else:
        confidence = 0.0

    # 🔹 DEBUG → terminal in VS Code
    print("\n==============================")
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] Raw scores: {scores}")
    print(f"[DEBUG] Confidence: {confidence:.3f}")
    print("==============================\n")

    context = "\n\n".join(
        f"{d.page_content}" for d in docs
    )

    prompt = f"""
Context:
{context}

Question: {query}

Answer:
- Answer in 8-15 sentences max
- Be concise and direct
- Write formulas in normal human-readable math (like a/b, x^2)
- Only use LaTeX if needed, wrapped in $...$
"""

    response = llm.invoke(prompt)

    # 🔹 OPTIONAL: append confidence to answer (can remove if you don't want UI display)
    return response.content + f"\n\n---\nConfidence: {confidence:.2f}"


# =======================
# RENDER MESSAGE (for maths)
# =======================

def render_message(content):
    # normalize bad latex formats
    content = content.replace("\\(", "$").replace("\\)", "$")
    content = content.replace("\\[", "$$").replace("\\]", "$$")

    st.markdown(content, unsafe_allow_html=True)


# =======================
# APP
# =======================

def main():

# ==============================================================================
    st.set_page_config(
        page_title="IALA Chat",
        layout="wide",
        initial_sidebar_state="expanded"
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

    with st.spinner("Loading knowledge..."):
        vector_store = build_vector_store()

    # CHAT HISTORY
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_message(msg["content"])

    # INPUT
    if prompt := st.chat_input("Ask a question..."):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = init_llm()
                answer = get_answer(prompt, vector_store, llm)
                render_message(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

