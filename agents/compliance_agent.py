"""
Compliance Agent — RBI Rule Adherence Checker
----------------------------------------------
Checks if applicant details follow the RBI guidelines retrieved by the RAG Agent.

Input:
  {
    "applicant": {...},
    "guidelines": {
        "feature_guidelines": {
            "age_years": {"summary": "..."},
            "ltv_ratio": {"summary": "..."},
            ...
        }
    }
  }

Output:
  {
    "compliance_summary": str,
    "violations": [
        {"feature": "age_years", "rule": "Minimum age 21 years", "status": "violation"},
        {"feature": "ltv_ratio", "rule": "LTV should not exceed 80%", "status": "compliant"}
    ],
    "compliance_score": float (0–1)
  }
"""

import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.llm_utility import get_llm

def check_rbi_compliance(applicant: dict, guidelines: dict) -> dict:
    """
    Uses an LLM to compare applicant features against RBI guideline summaries
    and identify compliance or violations.
    """
    llm = get_llm()
    prompt_template = PromptTemplate.from_template("""
You are an RBI compliance officer.
You are given applicant data and RBI guideline summaries per feature.

Your job: evaluate compliance only — do NOT write Python code.

For each feature:
- Decide if the applicant complies with the guideline.
- Add a short comment.
- Estimate an overall compliance_score (0–1, where 1 = fully compliant).

Return the result strictly as valid JSON in this format:

{{
  "violations": [
    {{"feature": "feature_name", "rule": "rule_text", "status": "compliant/violated", "comment": "reasoning"}},
    ...
  ],
  "compliance_score": 0.0,
  "compliance_summary": "short plain-text summary"
}}

Applicant Data:
{applicant_json}

RBI Guideline Summaries:
{guideline_json}
""")




    from langchain_core.output_parsers import StrOutputParser

    chain = prompt_template | llm | StrOutputParser()
    result = chain.invoke({"applicant": applicant, "guidelines": guidelines}).strip()


    # optional: try to extract valid JSON from the response
    json_block = re.search(r"\{[\s\S]*\}", result)
    if json_block:
        try:
            import json
            parsed = json.loads(json_block.group())
            return parsed
        except:
            pass

    # fallback if LLM didn’t format JSON
    return {
        "compliance_summary": result.strip(),
        "violations": [],
        "compliance_score": 0.0
    }

# ---------------------------------------------------------------------
# 🧪 TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    applicant = {
        "loan_type": "housing loan",
        "age_years": 19,
        "ltv_ratio": 0.9,
        "monthly_income_inr": 20000
    }

    guidelines = {
        "feature_guidelines": {
            "age_years": {"summary": "Applicant must be at least 21 years old."},
            "ltv_ratio": {"summary": "LTV should not exceed 80% for housing loans."}, 
            "monthly_income_inr": {"summary": "Minimum income requirement of ₹25,000 per month."}
        }
    }

    res = check_rbi_compliance(applicant, guidelines)
    print("\n=== Compliance Result ===")
    import json
    print(json.dumps(res, indent=2))
