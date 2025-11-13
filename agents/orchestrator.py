# agents/orchestrator.py
"""
Minimal CrewAI orchestrator:
- Tool: RBI Guideline Retriever  -> retrieve_rbi_guidelines_for_loan
- Tool: Risk Scorer              -> compute_risk_score
- Agent: RAGAgent                -> uses RBI Guideline Retriever
- Agent: RiskAgent               -> uses Risk Scorer
- Runner: executes RAG task then Risk task (sequential)
"""

import os
import sys
import json
from typing import Any, Dict

from typer import prompt

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import BaseModel, Field
from crewai import Crew, Agent, Task, LLM
from crewai.tools import tool
from agents.xai_agent import preprocess, model_predict, explain_prediction, _load_model_and_explainers
# Local modules
from agents.rag_agent import retrieve_feature_guidelines
from agents.risk_agent import compute_risk_score
from agents.compliance_agent import check_rbi_compliance
from utils.llm_utility import get_llm

# -----------------------------------------------------------------------------
# LLM CONFIG (Groq via OpenAI-compatible env vars; no LiteLLM required)
# -----------------------------------------------------------------------------


llm = LLM(
    model="openai/gpt-4o",
    temperature=0.2,
    max_tokens=6000,
    base_url="https://api.openai.com/v1",
)

# -----------------------------------------------------------------------------
# TOOL ARG SCHEMAS (make function-calling deterministic)
# -----------------------------------------------------------------------------
class RAGInput(BaseModel):
    applicant_raw: Dict[str, Any] = Field(..., description="Applicant JSON used to determine loan type and query policy.")

class RiskInput(BaseModel):
    applicant: Dict[str, Any] = Field(..., description="Applicant JSON for risk scoring.")

# -----------------------------------------------------------------------------
# TOOLS
# -----------------------------------------------------------------------------
# ...existing code...
# ...existing code...


# ---------------- TOOL 3 ----------------
@tool("XAI Reporter")
def xai_reporter_tool(applicant_data: dict,
                      compliance_data: dict,
                      risk_data: dict) -> dict:
    """
    Generates user and regulator-friendly explanations for a loan decision
    using LLM summarization of SHAP & LIME insights.
    """
    preprocessed = preprocess(applicant_data)
    output = explain_prediction(preprocessed, applicant_data, compliance_data, risk_data)
    return output


@tool("RBI Guideline Retriever")
def rag_agent_tool(applicant_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve RBI rules for the applicant's all features(for example "age_years", "monthly_income_inr", "foir_total_obligations_pct",
        "ltv_ratio", "pep_flag", "interest_type", "loan_type") from Chroma DB.
    Instrumented: prints entry/exit and returns 'error' on failure so orchestrator can surface issues.
    """
    print("[RAG TOOL] invoked with applicant_raw keys:", list(applicant_raw.keys()) if isinstance(applicant_raw, dict) else type(applicant_raw))
    try:
        res = retrieve_feature_guidelines(applicant_raw)
        out = {
            "loan_type": res.get("loan_type"),
            # The 'feature_guidelines' key is mandatory for the next agent to use the data
            "feature_guidelines": res.get("feature_guidelines", {}) 
        }

        print(f"[RAG TOOL] output: {out['feature_guidelines'].keys()}")

        # Assuming we can determine the total number of retrieved chunks from the nested dict for printing:
        total_chunks = sum(len(d.get("retrieved_guidelines", [])) for d in out["feature_guidelines"].values())
        
        print(f"[RAG TOOL] success — retrieved {total_chunks} total guideline chunks across features")

        return out
    except FileNotFoundError as fnf:
        msg = str(fnf)
        print("[RAG TOOL] FileNotFoundError:", msg)
        return {"loan_type": None, "query": "", "retrieved_guidelines": [], "error": msg}
    except ValueError as ve:
        msg = str(ve)
        print("[RAG TOOL] ValueError:", msg)
        return {"loan_type": None, "query": "", "retrieved_guidelines": [], "error": msg}
    except Exception as e:
        msg = repr(e)
        print("[RAG TOOL] Exception:", msg)
        return {"loan_type": None, "query": "", "retrieved_guidelines": [], "error": msg}


@tool("Risk Scorer")
def risk_scorer_tool(applicant: dict[str, Any])-> Dict[str, Any]:
    """
    Compute explainable risk score (0–10, higher = riskier) for an application.
    Output includes: risk_score_10, grade, components, weights, reasons, drivers, context
    """
    print("[RISK TOOL] invoked with applicant keys:", list(applicant.keys()) if isinstance(applicant, dict) else type(applicant))
    try:
        score_obj = compute_risk_score(applicant)
        # Expect compute_risk_score to return a dict with at least 'risk_score_10' and 'grade'
        out = {
            "risk_score_10": score_obj.get("risk_score_10"),
            "grade": score_obj.get("grade"),
            "components": score_obj.get("components", {}),
            "drivers": score_obj.get("drivers", []),
            "context": score_obj.get("context", {}),
        }
        print(f"[RISK TOOL] success — score: {out['risk_score_10']} grade: {out['grade']}")
        return out
    except Exception as e:
        msg = repr(e)
        print("[RISK TOOL] Exception:", msg)
        return {"risk_score_10": None, "grade": None, "components": {}, "drivers": [], "context": {}, "error": msg}
    
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser



@tool("Compliance Checker")
def compliance_checker_tool(applicant: Dict[str, Any], guidelines: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses an LLM to analyze applicant details against RBI guideline summaries.
    Returns a structured JSON with feature-wise compliance and a final verdict.
    """

    llm = get_llm()

    prompt = PromptTemplate.from_template("""
You are an RBI compliance officer.
Compare each feature in the applicant's data with the guideline summaries provided below.

Rules:
- "is_compliant": true if applicant follows the RBI rule
- "is_compliant": false if violates
- "is_compliant": "NO_DATA" if guideline is unclear or missing
- Give a short, clear "reason" for each feature
- final_verdict:
    - "FAIL" if any feature violates
    - "PASS" if all are compliant
    - "INCOMPLETE" if rules missing for most features

Return STRICTLY in valid JSON (no markdown, no explanations):

{{
  "feature_compliance": {{
    "<feature_name>": {{
      "is_compliant": true | false | "NO_DATA",
      "reason": "<reason>"
    }}
  }},
  "final_verdict": "PASS" | "FAIL" | "INCOMPLETE"
}}

### Applicant JSON:
{applicant}

### RBI Guidelines JSON:
{guidelines}
""")


    # safer chain (no deprecated LLMChain)
    from langchain.schema.runnable import RunnableSequence
    from langchain.schema.output_parser import StrOutputParser

    chain = prompt | get_llm() | StrOutputParser()


    try:
        raw_output = chain.invoke({"applicant": applicant, "guidelines": guidelines}).strip()

        # ✅ enforce JSON parsing safety
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            # fallback: try to clean accidental markdown / bad chars
            cleaned = raw_output.strip("` \n").replace("json", "")
            parsed = json.loads(cleaned)

        # ✅ enforce structure if model drops keys
        if "feature_compliance" not in parsed:
            parsed["feature_compliance"] = {}
        if "final_verdict" not in parsed:
            parsed["final_verdict"] = "INCOMPLETE"

        return parsed

    except Exception as e:
        return {
            "error": f"Compliance checker failed: {str(e)}",
            "feature_compliance": {},
            "final_verdict": "INCOMPLETE"
        }
@tool("Decision Evaluator")
def decision_evaluator_tool(policy_output: dict, risk_output: dict) -> dict:
    """
    Determines final loan eligibility, and if approved, selects ML model type.
    Returns structured reasoning, probability, and final decision.
    """

    prompt = PromptTemplate.from_template("""
You are an RBI-regulated credit decision AI.

Your task is to make a *two-step decision* based on the applicant's RBI compliance and credit risk profile.

---

### 🧩 Stage 1: Loan Eligibility
1. Use **PolicyAgent Output** to determine compliance:
   - If `final_decision` == "DECLINED" or any critical violations exist → reject immediately.
   - If `final_decision` == "APPROVED" → continue to risk evaluation.
2. Use **RiskAgent Output** to adjust eligibility:
   - If `risk_score_10` > 7 → high risk → reject with low probability (≈0.3–0.4)
   - If 4 ≤ `risk_score_10` ≤ 7 → moderate risk → cautious approval (≈0.6–0.75)
   - If `risk_score_10` < 4 → low risk → confident approval (≈0.85–0.95)
3. Give reasoning for rejection if ineligible (e.g., non-compliance, excessive risk, missing data).

---

### ⚙️ Stage 2: Model Selection (only if loan_eligible = true)
- Choose model type:
  - `"blackbox"` → for borderline/moderate risk where nonlinear interactions likely matter.
  - `"interpretable"` → for low-risk applicants needing explainable audit-friendly scoring.
- Provide justification for your chosen model.

---

### 🎯 Output Format (STRICT JSON only)
{{
  "loan_eligible": true | false,
  "decision_reason": "<clear 2–3 line reasoning>",
  "selected_model": "blackbox" | "interpretable" | "none"
}}

### PolicyAgent Output:
{policy_output}

### RiskAgent Output:
{risk_output}
""")

    
    from langchain.schema.output_parser import StrOutputParser
   
    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "policy_output": policy_output,
            "risk_output": risk_output
        })
        if not result or not result.strip():
            raise ValueError("Empty response from LLM")
    except Exception as e:
        print(f"[RISK AGENT ERROR] {e}")
        result = json.dumps({
            "risk_score": 0.0,
            "risk_grade": "UNKNOWN",
            "reason": str(e)
        })



        # ensure valid structure
        if not isinstance(result, dict):
            result = json.loads(result)
        result.setdefault("loan_eligible", False)
        result.setdefault("selected_model", "none")
        result.setdefault("decision_reason", "No reasoning provided.")

        return result

    except Exception as e:
        return {
            "error": f"Decision evaluator failed: {e}",
            "loan_eligible": False,
            "selected_model": "none",
            "decision_reason": f"Decision process error: {str(e)}"
        }
@tool("explainable AI Tool")
def xai_tool(applicant: Dict[str, Any], risk_output: Dict[str, Any], compliance_output: Dict[str, Any],decision_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provides clear, concise explanations for risk scores and compliance decisions to ensure transparency.
    """

    llm = get_llm()

    prompt_template = PromptTemplate.from_template("""

    """)


# -----------------------------------------------------------------------------
# AGENTS (single-tool, single-iteration to avoid repeated calls)
# -----------------------------------------------------------------------------
policy_agent = Agent(
    name="PolicyAgent",
    role="RBI Policy Validator",
    goal=(
        "You are an RBI compliance expert. Your job is to: "
        "1️⃣ Use the 'RBI Guideline Retriever' to fetch guidelines for each feature. "
        "2️⃣ Then use the 'Compliance Checker' to validate the applicant. "
        "ALWAYS use this format exactly:\n\n"
        "Thought: explain what you will do\n"
        "Action: choose exactly one tool name\n"
        "Action Input: valid JSON of inputs to that tool\n"
        "Observation: tool output\n\n"
        "Once done, give your final structured JSON as:\n"
        "Final Answer: {\n"
        "  'retrieved_guidelines': {...},\n"
        "  'feature_compliance': {...},\n"
        "  'final_verdict': 'PASS'|'FAIL'|'INCOMPLETE'\n"
        "}\n\n"
        "Do NOT continue thinking after that or use any more tools."
    ),
    backstory="An expert compliance officer ensuring all loan applications follow RBI rules before approval.",
    tools=[rag_agent_tool, compliance_checker_tool],
    verbose=True,
    max_iter=2,
    llm=llm,
    allow_delegation=False,
    always_use_tools=True,  # 👈 keep this here
)

decision_agent = Agent(
    name="DecisionAgent",
    role="Final Loan Decision Maker",
    goal=(
        "Analyze the compliance and risk reports in two stages: "
        "first determine whether the applicant is eligible for loan approval, "
        "and if approved, decide which model type (blackbox or interpretable) should be used."
    ),
    backstory=(
        "A senior credit AI officer ensuring that all loan decisions align with RBI compliance, "
        "risk assessment, and model governance best practices."
    ),
    context=[],
    tools=[decision_evaluator_tool],
    llm=llm,
    verbose=True,
    max_iter=3,
    allow_delegation=False,
    always_use_tools=True,
)

risk_agent = Agent(
    name="RiskAgent",
    role="Risk Scoring Analyst",
    goal="Return the risk score for the applicant using the Risk Scorer tool only — no reasoning or multiple actions.",
    backstory="Credit risk specialist that produces quantitative risk scores.",
    tools=[risk_scorer_tool],
    verbose=True,
    max_iter=1,
    llm=llm,
    allow_delegation=False,
    always_use_tools=True
)

xai_agent = Agent(
    name="XAIAgent",
    role="Explainable AI Officer",
    goal=(
        "Provide explainable credit model reasoning. "
        "First predict using the trained model, "
        "then explain the outcome using SHAP & LIME, "
        "and finally summarize results for both user and regulator."
    ),
    backstory=(
        "An AI explainability auditor ensuring every loan decision "
        "is justified with interpretable insights and regulatory clarity."
    ),
    llm=llm,
    tools=[xai_reporter_tool],
    verbose=True,
    max_iter=3,
    allow_delegation=False,
    always_use_tools=True,
)

# -----------------------------------------------------------------------------
# RUNNER (no fallbacks)
# -----------------------------------------------------------------------------
def run_sync(applicant_dict: Dict[str, Any]):
    """
    Runs two tasks sequentially (no fallbacks):
      1) RAGAgent -> RBI Guideline Retriever
      2) RiskAgent -> Risk Scorer
    Returns CrewAI's raw result.
    """
    policy_task = Task(
        description=(
            "You are an RBI compliance expert. "
            "Retrieve RBI guidelines using the 'RBI Guideline Retriever' tool, "
            "then check the applicant's compliance using the 'Compliance Checker' tool. "
            "If any rule is violated, clearly mark the application as DECLINED. "
            "Return a final JSON with guideline summaries, compliance_score, violations, and final_decision."
        ),
        expected_output=(
            "{'retrieved_guidelines': dict, 'violations': list, "
            "'compliance_score': float, 'final_decision': 'APPROVED'|'DECLINED'}"
        ),
        agent=policy_agent,
    )

    risk_task = Task(
        description=(
            "Call the tool 'Risk Scorer' using the context variable 'applicant'. "
            "Return ONLY the tool JSON with keys: risk_score_10 and grade (plus details)."
        ),
        expected_output="JSON with keys: risk_score_10, grade",
        agent=risk_agent,
    )
    
    decision_task = Task(
    description="Analyze outputs from PolicyAgent and RiskAgent to decide loan eligibility and appropriate model type.",
    expected_output=(
        "JSON with keys: loan_eligible, approval_probability, decision_reason, selected_model"
    ),
    agent=decision_agent
)
    xai_task = Task(
        description=(
            "Generate clear explanations for the loan decision using the 'XAI Reporter' tool. "
            "Incorporate insights from compliance and risk analyses."
        ),
        expected_output="JSON with keys: user_explanation, regulator_explanation",
        agent=xai_agent,
    )
    
    crew = Crew(
        name="LoanDecisionCrew",
        agents=[policy_agent,risk_agent,decision_agent,xai_agent],
        tasks=[policy_task,risk_task,decision_task,xai_task],
        process="sequential",
        llm=llm,
        verbose=False,
    )

    # Provide both keys in context so each tool can be called unambiguously
    inputs = {"applicant_raw": applicant_dict, "applicant": applicant_dict}
    result = crew.kickoff(inputs={"applicant_raw": applicant_dict, "applicant": applicant_dict})

    return result



# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "application_id": "APP-0001",
        "loan_type": "housing loan",
        "age_years": 32,
        "bureau_score": 720,
        "monthly_income_inr": 55000,
        "foir_total_obligations_pct": 25.0,
        "requested_amount_inr": 300000,
        "tenure_months": 36,
        "gender": "Male",
        "state": "Maharashtra",
        "kyc_mode": "Video KYC",
        "ovd_type": "Aadhaar",
        "interest_type": "Fixed",
        "pep_flag": False,
        "kfs_provided": True,
        "processing_fee_inr": 2000.0,
        "other_charges_inr": 500.0,
        "apr_pct": 12.5,
        "property_value_inr": 1000000,
        "ltv_ratio": 0.3,
    }

    result = run_sync(sample)
    print("\n=== Crew Result ===")
    print(json.dumps(result, indent=2, default=str))
