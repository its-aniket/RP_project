from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.llm_utility import get_llm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os, re
import warnings
from langchain_core._api import LangChainDeprecationWarning  # ✅ correct warning class

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = os.path.join("data", "embeddings")
COLLECTION_NAME = "rbi_guidelines"

llm = get_llm()

# ---------------------------------------------------------------------
# 🧠 Smart Summarizer (LLM + Heuristics)
# ---------------------------------------------------------------------
def clean_guideline_text(text: str) -> str:
    """Lightly clean text but keep enough for context."""
    if not text or len(text.strip()) < 30:
        return "No direct RBI rule found."

    # remove irrelevant filler
    text = re.sub(r"(no direct rbi rule|unable to find|not mentioned|no explicit mention).*?\n", "", text, flags=re.I)
    # collapse whitespace and limit
    text = re.sub(r"\s+", " ", text).strip()[:2000]
    return text


def summarize_guidelines(raw_texts: list[str]) -> str:
    """Condense retrieved RBI text into short, meaningful rule summaries."""
    if not raw_texts:
        return "No direct RBI rule found."

    text = " ".join(raw_texts)
    cleaned = clean_guideline_text(text)
    if "No direct RBI rule found" in cleaned:
        return "No direct RBI rule found."

    summarizer_prompt = PromptTemplate.from_template("""
    You are an RBI policy expert.
    From the following RBI policy text, extract specific *numeric* or *rule-based* limits,
    eligibility conditions, or thresholds (e.g., "LTV ≤ 80%", "Age ≥ 21 years", "Income ≥ ₹25,000").
    Include 1–3 key rules in bullet format. If none found, respond with: "No direct RBI rule found."

    RBI Text:
    {text}

    Summary:
    """)

    chain = LLMChain(llm=llm, prompt=summarizer_prompt)
    try:
        summary = chain.run({"text": cleaned}).strip()
    except Exception:
        return cleaned

    # 🔍 post-trim any “please paste text” or irrelevant replies
    if re.search(r"paste the text|unable to access", summary, flags=re.I):
        return "No direct RBI rule found."

    # shorten if too verbose
    summary_lines = [l.strip("•* ") for l in summary.splitlines() if l.strip()]
    summary = " ".join(summary_lines[:3])
    return summary or "No direct RBI rule found."

# ---------------------------------------------------------------------
# 🔍 RAG Retriever + Summarizer
# ---------------------------------------------------------------------
def retrieve_feature_guidelines(applicant_data: dict, top_k: int =3) -> dict:
    """
    For each relevant applicant feature, retrieve RBI guideline snippets,
    summarize, and return only concise policy points.
    """
    embedder = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
    )

    # Only meaningful regulatory features
    relevant_features = [
        "age_years", "monthly_income_inr", "foir_total_obligations_pct",
        "ltv_ratio", "pep_flag", "interest_type", "loan_type"
    ]

    results = {}
    for feature in relevant_features:
        value = applicant_data.get(feature)
        if value is None:
            continue

        clean_feature = re.sub(r"_", " ", feature)
        query = f"RBI guideline related to {clean_feature} {value} for {applicant_data.get('loan_type', 'loan')}"

        # retrieve top-k guideline chunks
        retrieved_docs = db.similarity_search(query, k=top_k)
        retrieved_texts = [r.page_content for r in retrieved_docs]

        # summarize or return default
        summary = summarize_guidelines(retrieved_texts)

        results[feature] = {
            "value": value,
            "query": query,
            "retrieved_guidelines": retrieved_texts[:1],  # keep first chunk only
            "summary": summary
        }

    return {
        "loan_type": applicant_data.get("loan_type"),
        "feature_guidelines": results
    }


# ---------------------------------------------------------------------
# 🧪 Test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    applicant = {
        "loan_type": "housing loan",
        "age_years": 32,
        "monthly_income_inr": 55000,
        "foir_total_obligations_pct": 25.0,
        "ltv_ratio": 0.3,
        "pep_flag": False,
        "interest_type": "Fixed"
    }

    result = retrieve_feature_guidelines(applicant)
    print(result)
