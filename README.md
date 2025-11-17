# 📘 Agentic Explainable AI Framework for Loan Approval

## 🧠 Project Overview  
This project implements an **Agentic Explainable AI (XAI) Framework** for loan approval that combines:

- **Regulatory Compliance (RBI guidelines)**  
- **Risk Assessment**  
- **Dynamic Model Selection**  
- **Explainable Decisions (SHAP & LIME)**  

The system uses **CrewAI agents**, **XGBoost**, and **Neural Networks**, ensuring every decision is **accurate, fair, transparent, and regulator-ready**.

---

## 🎯 Objectives  
- Automate loan approval using AI-driven agents.  
- Quantify borrower risk using credit, income, and loan attributes.  
- Explain model decisions using SHAP and LIME.  
- Ensure regulatory compliance and fairness (RBI standards).  
- Compare model interpretability vs profitability.

---

## 📊 Dataset Description  

**File:** `loan_approval_model.xlsx`  
**Rows:** 40,000  
**Columns:** 26  

### **Feature Groups**

| Category | Columns | Description |
|---------|---------|-------------|
| **Applicant Demographics** | `age_years`, `gender_Female`, `gender_Male`, `gender_Other`, `pep_flag` | Applicant attributes and compliance flags. |
| **Income & Obligations** | `monthly_income_inr`, `existing_monthly_obligations_inr`, `foir_total_obligations_pct` | Income strength and existing liabilities. |
| **Loan Details** | `requested_amount_inr`, `sanctioned_amount_inr`, `tenure_months`, `interest_rate_annual_pct`, `processing_fee_inr`, `other_charges_inr`, `apr_pct`, `proposed_emi_inr` | Loan request information and total loan cost. |
| **Credit Behavior** | `bureau_score`, `ltv_ratio`, `ovd_provided` | Credit score, loan-to-value ratio, and documentation. |
| **Property & Application** | `property_value_inr`, `application_month`, `interest_type_encoded`, `pin_code` | Property value, region, and loan context. |
| **Process Variables** | `time_to_sanction_days`, `kfs_provided` | Operational and compliance checkpoints. |
| **Target Variable** | `target` | Loan approved (1) or rejected (0). |

✔ No missing values  
✔ All numeric fields  

---

## ⚙️ Machine Learning Models  

| Model | Type | Purpose |
|--------|------|---------|
| **XGBoost** | Black-box | Highest accuracy, good profitability. |
| **Neural Network (MLP)** | Black-box | Captures nonlinear interactions. |
| **Logistic Regression** | Interpretable | Used when transparency is required. |

---

## 🤖 Agentic AI Architecture  

This project uses **four agents** orchestrated via **CrewAI**.

### **1. Policy Agent**
- Fetches RBI guidelines.  
- Ensures:  
  - `pep_flag` handling  
  - `foir_total_obligations_pct` ≤ permitted limit  
  - `kfs_provided` must be 1  
- Ensures fairness (no discrimination by gender or PIN).  
- Output: **Policy Compliance Report**

---

### **2. Risk Agent**
- Computes **risk_score (0–1)** using:  
  - `bureau_score`  
  - `ltv_ratio`  
  - `monthly_income_inr`  
  - `existing_monthly_obligations_inr`  
- If risk_score > 0.7 → High-risk → Review required  
- Else → Send to Decision Agent  
- Output: **Risk Score + Category**

---

### **3. Decision Agent**
- Checks Policy Agent + Risk Agent outputs  
- Chooses model based on context:  
  - **Logistic Regression** → when interpretability is priority  
  - **XGBoost / Neural Network** → when high accuracy/profit required  
- Generates:  
  - Approve / Reject / Review  
  - Confidence score  
- Output: **Decision + Selected Model**

---

### **4. XAI Agent**
- Runs SHAP (global + local explanations)  
- Runs LIME for case-specific reasoning  
- Generates:  
  - **Customer Report** (simple language)  
  - **Regulator Report** (SHAP values, fairness & policy compliance)  
- Output: **Full Explainability Package**

---

## 🔄 End-to-End Workflow  

```
1. Policy Agent → Regulatory compliance check  
2. Risk Agent → Credit risk estimation  
3. Decision Agent → Model selection + decision  
4. XAI Agent → Transparent explanations  
```

This workflow ensures **Responsible AI aligned with RBI model governance and fairness**.

---

## 🧾 Example Outputs  

```
Policy Agent: Compliant – FOIR = 75%, KFS Provided = Yes  
Risk Agent: Risk Score = 0.42 → Medium Risk  
Decision Agent: Model Used: XGBoost; Decision: Approved; Confidence = 0.88  
XAI Agent: Top SHAP Features → bureau_score (+0.27), ltv_ratio (–0.18), monthly_income_inr (+0.14)
```

---

## 💻 Tech Stack  
- Python 3.10  
- CrewAI  
- scikit-learn  
- XGBoost  
- Neural Networks (TensorFlow/PyTorch)  
- SHAP, LIME  
- Google Colab (Free Tier)

---

## 📁 Project Structure  

```text
.
├── agents/          # CrewAI agent definitions (Policy, Risk, Decision, XAI)
├── api/             # Backend services (FastAPI / Flask for serving predictions & explanations)
├── frontend/        # UI components (dashboard / web interface)
├── models/          # Saved model binaries & training utilities
├── notebooks/       # Jupyter/Colab notebooks (EDA, model training, SHAP analysis)
├── pipeline/        # End-to-end workflow scripts & orchestration
├── research/        # Literature review, experimental logs, academic notes
├── rules/           # RBI policy snippets, fairness rules, and agent prompt templates
├── tests/           # Unit tests & integration tests for agents, API, and pipeline
├── utils/           # Helper functions (preprocessing, common utilities)
├── .gitignore
├── README.md
├── config.yaml      # Global configuration (paths, thresholds, model configs)
└── requirements.txt # Python dependencies


```

---

## 🧩 Key Contributions  
- RBI-aligned compliance validation  
- Full agent-driven workflow (Policy → Risk → Decision → XAI)  
- SHAP + LIME transparency  
- Dynamic model selection  
- Strong interpretability vs profitability analysis  

---

## 🚀 Future Scope  
- Integrate Fraud Detection & Insurance Pricing  
- Add real-time ingestion (CKYC, Bureau APIs)  
- Deploy as Streamlit dashboard  
- Add RAG for policy retrieval  
- Reinforcement Learning for agent optimization  

---

## 👥 Team  
- **Vedant Bhave** – Agentic AI Pipeline, Risk Modeling  
- **Aniket Jadhav** – Model Training, Explainability, Evaluation  
