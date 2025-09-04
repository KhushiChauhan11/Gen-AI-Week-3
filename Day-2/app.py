from pathlib import Path
import streamlit as st
import os

# ----------------------------
# Data setup
# ----------------------------
DATA_DIR = Path.cwd() / "rag_data"
DATA_DIR.mkdir(exist_ok=True)

salary_path = DATA_DIR / "salary.txt"
insurance_path = DATA_DIR / "insurance.txt"

if not salary_path.exists():
    salary_path.write_text(
        "Salary structure:\n- Monthly salary is the fixed amount paid every month.\n"
        "- Annual salary = monthly salary * 12.\n- Deductions include taxes, PF, professional tax.\n- Net salary = gross - deductions.\n",
        encoding="utf-8",
    )

if not insurance_path.exists():
    insurance_path.write_text(
        "Insurance benefits:\n- Covers specified medical treatments and hospitalisation.\n"
        "- Premium is paid monthly or yearly.\n- Claim process: notify insurer, submit claim + bills, insurer assesses claim.\n",
        encoding="utf-8",
    )

st.set_page_config(page_title="Multi-Agent RAG", layout="wide")
st.title("🚀 Multi-Agent RAG — Salary & Insurance")

# ----------------------------
# Try Groq LLM
# ----------------------------
USE_GROQ = False
try:
    from langchain_groq import ChatGroq
    groq_key = st.secrets.get("GROQ_API_KEY")
    if groq_key:
        llm = ChatGroq(groq_api_key=groq_key, model="llama-3.1-8b-instant")
        USE_GROQ = True
except Exception:
    USE_GROQ = False

# ----------------------------
# Offline TF-IDF fallback
# ----------------------------
if not USE_GROQ:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    DOCS = []
    METADATA = []
    for p in sorted(DATA_DIR.glob("*.txt")):
        txt = p.read_text(encoding="utf-8")
        DOCS.append(txt)
        tag = "salary" if "salary" in p.name.lower() else "insurance"
        METADATA.append({"source": p.name, "domain": tag})

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(DOCS)

    def retrieve_tfidf(query, top_k=2):
        qv = vectorizer.transform([query])
        sims = cosine_similarity(qv, tfidf_matrix)[0]
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idxs]

    def salary_agent(query):
        results = retrieve_tfidf(query)
        salary_results = [r for r in results if METADATA[r[0]]["domain"] == "salary"]
        if not salary_results:
            return "I couldn't find salary info."
        return DOCS[salary_results[0][0]]

    def insurance_agent(query):
        results = retrieve_tfidf(query)
        ins_results = [r for r in results if METADATA[r[0]]["domain"] == "insurance"]
        if not ins_results:
            return "I couldn't find insurance info."
        return DOCS[ins_results[0][0]]

# ----------------------------
# Coordinator Agent
# ----------------------------
SALARY_KW = {"salary", "annual", "monthly", "gross", "net", "deduction", "pf", "hra", "pay"}
INSURANCE_KW = {"insurance", "premium", "claim", "coverage", "policy", "hospital"}

def coordinator(query):
    ql = query.lower()
    sal_hits = sum(1 for k in SALARY_KW if k in ql)
    ins_hits = sum(1 for k in INSURANCE_KW if k in ql)
    agent = "salary" if sal_hits >= ins_hits else "insurance"

    if USE_GROQ:
        if agent == "salary":
            return llm.invoke(query).content
        else:
            return llm.invoke(query).content
    else:
        if agent == "salary":
            return salary_agent(query)
        else:
            return insurance_agent(query)

# ----------------------------
# Streamlit UI
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Chat")
    user_q = st.text_input("Ask a question:")
    if st.button("Send"):
        if user_q.strip():
            st.session_state.history.append(("You", user_q.strip()))
            ans = coordinator(user_q.strip())
            st.session_state.history.append(("Assistant", ans))
        else:
            st.warning("Please enter a query!")

    # Example queries
    if st.button("Example: Annual Salary?"):
        q = "How do I calculate annual salary?"
        st.session_state.history.append(("You", q))
        st.session_state.history.append(("Assistant", coordinator(q)))

    if st.button("Example: Insurance Policy?"):
        q = "What is included in my insurance policy?"
        st.session_state.history.append(("You", q))
        st.session_state.history.append(("Assistant", coordinator(q)))

    # Display chat history
    for role, text in st.session_state.history:
        st.markdown(f"**{role}:** {text}")

with col2:
    st.subheader("Info / Debug")
    st.write("Mode:", "Groq LLM" if USE_GROQ else "Offline TF-IDF")
    st.write("Data files:")
    for p in sorted(DATA_DIR.glob("*.txt")):
        st.write("-", p.name)
