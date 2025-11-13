# risk_agent.py
from typing import Dict, Any

def compute_risk_score(
    applicant: Dict[str, Any],
    policy: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Compute an explainable risk score on 0–10 (10 = highest risk).
    Works across housing / gold / vehicle / personal / microfinance / msme.

    Inputs
    ------
    applicant : dict
        Example keys (robust to missing):
          - loan_type (str) e.g., "housing", "gold", "vehicle", "personal", "microfinance", "msme"
          - bureau_score (int: 300–900)
          - foir_total_obligations_pct (float: 0–100)
          - ltv_ratio (float: 0–1)  # for secured loans
          - interest_rate_annual_pct (float)
          - pep_flag (bool)
          - age_years, monthly_income_inr, etc. (optional)

    policy : dict | None
        Policy/benchmark params. If None, uses defaults baked below.

    Returns
    -------
    dict with:
      - risk_score_10 (float)     # final score 0..10, higher = riskier
      - grade (str)               # A+..E band
      - components (dict)         # each component risk in 0..1
      - weights (dict)            # weights used
      - reasons (list[str])       # short reason codes/messages
    """
    # -------------------------
    # Helpers
    # -------------------------
    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return hi if x > hi else (lo if x < lo else x)

    def norm_bureau(b: float) -> float:
        # Risk↑ as bureau↓. 800+ => ~0 risk; 600 or below => ~1 risk.
        if b is None:
            return 0.5
        return clamp((800 - b) / 200)  # 800→0, 600→1

    def norm_foir(foir_pct: float) -> float:
        # Risk↑ as FOIR↑. <=30% → 0; >=65% → 1
        if foir_pct is None:
            return 0.5
        return clamp((foir_pct - 30.0) / 35.0)

    def norm_rate(rate: float, mean: float) -> float:
        # Risk↑ as rate exceeds reference mean. +20% above mean → 1
        if rate is None or mean is None:
            return 0.3
        return clamp((rate - mean) / max(0.001, (0.20 * mean)))

    def norm_ltv(ltv: float, cap: float) -> float:
        # Risk↑ as LTV approaches/exceeds cap.
        # 80% of cap → 0 risk; at cap → ~1; beyond cap → clipped to 1
        if ltv is None or cap is None or cap <= 0:
            return 0.0
        rel = ltv / cap
        return clamp((rel - 0.80) / 0.20)

    def grade_from_score(s10: float) -> str:
        # Safer bands at lower risk
        if s10 <= 2.0:  return "A+"
        if s10 <= 3.5:  return "A"
        if s10 <= 5.0:  return "B"
        if s10 <= 6.5:  return "C"
        if s10 <= 8.0:  return "D"
        return "E"

    # -------------------------
    # Defaults (override by passing policy)
    # -------------------------
    DEFAULT_POLICY = {
        "products": {
            "housing":      {"LTV_CAP": {"<=2_000_000": 0.90, "2_000_001-7_500_000": 0.80, ">7_500_000": 0.75}, "FOIR_MAX": 0.55, "RATE_MEAN": 9.0},
            "gold":         {"LTV_CAP": 0.75, "FOIR_MAX": 0.55, "RATE_MEAN": 13.5},
            "vehicle":      {"LTV_CAP": 0.90, "FOIR_MAX": 0.55, "RATE_MEAN": 10.5},
            "personal":     {"FOIR_MAX": 0.50, "RATE_MEAN": 14.0},
            "microfinance": {"FOIR_MAX": 0.50, "RATE_MEAN": 20.0},
            "msme":         {"FOIR_MAX": 0.55, "RATE_MEAN": 11.0},
        },
        # Base weights (sum ≈ 1 per product after pruning missing components)
        "weights": {
            "secured":  {"bureau": 0.35, "foir": 0.30, "ltv": 0.20, "rate": 0.10, "pep": 0.05},
            "unsecured":{"bureau": 0.40, "foir": 0.35, "rate": 0.20, "pep": 0.05},
        }
    }
    P = policy or DEFAULT_POLICY

    # -------------------------
    # Read applicant safely
    # -------------------------
    lt_raw = str(applicant.get("loan_type", "")).lower().strip()
    # normalize a few variants
    if "home" in lt_raw or "housing" in lt_raw:
        loan_type = "housing"
    elif "gold" in lt_raw:
        loan_type = "gold"
    elif "veh" in lt_raw or "auto" in lt_raw:
        loan_type = "vehicle"
    elif "micro" in lt_raw:
        loan_type = "microfinance"
    elif "msme" in lt_raw or "sme" in lt_raw:
        loan_type = "msme"
    else:
        loan_type = "personal"  # default to unsecured

    bureau   = applicant.get("bureau_score", None)
    foir_pct = applicant.get("foir_total_obligations_pct", None)  # 0..100
    ltv      = applicant.get("ltv_ratio", None)                   # 0..1
    rate     = applicant.get("interest_rate_annual_pct", None)
    pep      = bool(applicant.get("pep_flag", False))

    # Determine product references
    prod = P["products"].get(loan_type, {})
    ref_rate_mean = prod.get("RATE_MEAN", 12.0)

    # LTV cap resolution
    ltv_cap = None
    if loan_type in ("housing", "gold", "vehicle"):
        cap_def = prod.get("LTV_CAP")
        if isinstance(cap_def, dict) and "LTV_CAP" not in cap_def:
            # housing banded caps → approximate by property_value if present
            pv = applicant.get("property_value_inr", None)
            if pv is None:
                # fallback to mid-band cap
                ltv_cap = 0.80
            else:
                try:
                    pv = float(pv)
                    if pv <= 2_000_000:
                        ltv_cap = cap_def.get("<=2_000_000", 0.90)
                    elif pv <= 7_500_000:
                        ltv_cap = cap_def.get("2_000_001-7_500_000", 0.80)
                    else:
                        ltv_cap = cap_def.get(">7_500_000", 0.75)
                except Exception:
                    ltv_cap = 0.80
        elif isinstance(cap_def, (float, int)):
            ltv_cap = float(cap_def)

    # -------------------------
    # Component risks (0..1 each)
    # -------------------------
    r_bureau = norm_bureau(float(bureau)) if bureau is not None else 0.5
    r_foir   = norm_foir(float(foir_pct)) if foir_pct is not None else 0.5
    r_rate   = norm_rate(float(rate), float(ref_rate_mean)) if rate is not None else 0.3
    r_ltv    = norm_ltv(float(ltv), float(ltv_cap)) if (ltv is not None and ltv_cap is not None) else 0.0
    r_pep    = 1.0 if pep else 0.0  # PEP always increases risk fully on this component

    components = {
        "bureau": r_bureau,
        "foir":   r_foir,
        "rate":   r_rate,
        "ltv":    r_ltv,
        "pep":    r_pep,
    }

    # -------------------------
    # Weights (secured vs unsecured)
    # -------------------------
    secured = loan_type in ("housing", "gold", "vehicle")
    base_w  = P["weights"]["secured" if secured else "unsecured"].copy()

    # Drop weight if component not applicable, then renormalize
    if not secured:
        base_w.pop("ltv", None)
    # If no PEP flag provided, keep weight (defaults to False already handled)

    # Normalize weights to sum=1
    total_w = sum(base_w.values())
    weights = {k: v / total_w for k, v in base_w.items()} if total_w > 0 else base_w

    # -------------------------
    # Aggregate risk (0..1) → 0..10
    # -------------------------
    agg = 0.0
    for k, w in weights.items():
        agg += w * components[k]

    risk_score_10 = round(agg * 10.0, 2)
    grade = grade_from_score(risk_score_10)

    # -------------------------
    # Reasons (simple, human-readable)
    # -------------------------
    reasons = []
    if bureau is not None and bureau < 650:
        reasons.append(f"Low bureau score ({bureau})")
    if foir_pct is not None and foir_pct > 55:
        reasons.append(f"High FOIR ({foir_pct:.1f}%)")
    if ltv is not None and ltv_cap is not None and ltv >= 0.95 * ltv_cap:
        reasons.append(f"LTV near cap ({ltv:.2f} vs cap {ltv_cap:.2f})")
    if rate is not None and rate > 1.10 * ref_rate_mean:
        reasons.append(f"Rate above segment mean ({rate:.2f}% > {ref_rate_mean:.2f}%)")
    if pep:
        reasons.append("PEP flag present (EDD required)")

    # Always include top drivers by contribution (w * component)
    contrib_sorted = sorted(((k, weights.get(k, 0)*components[k]) for k in components if k in weights),
                            key=lambda t: t[1], reverse=True)
    top_drivers = [f"driver:{k} contrib:{c:.3f}" for k, c in contrib_sorted[:3]]

    return {
        "risk_score_10": risk_score_10,
        "grade": grade,
        "components": components,
        "weights": weights,
        "reasons": reasons or ["No major risk flags; driven by baseline factors."],
        "drivers": top_drivers,
        "context": {
            "loan_type": loan_type,
            "refs": {"rate_mean": ref_rate_mean, "ltv_cap": ltv_cap}
        }
    }



if __name__ == "__main__":
    allowed_grades = {"A+", "A", "B", "C", "D", "E"}

    samples = [
        {
            "name": "Housing-safe",
            "data": {
                "loan_type": "housing",
                "bureau_score": 780,
                "foir_total_obligations_pct": 32.0,
                "ltv_ratio": 0.72,
                "interest_rate_annual_pct": 9.1,
                "pep_flag": False,
                "property_value_inr": 3_500_000,
            },
            "secured": True,
        },
        {
            "name": "Personal-risky",
            "data": {
                "loan_type": "personal",
                "bureau_score": 610,
                "foir_total_obligations_pct": 64.0,
                "interest_rate_annual_pct": 17.2,
                "pep_flag": True,
            },
            "secured": False,
        },
    ]

    for case in samples:
        res = compute_risk_score(case["data"])
        score = res.get("risk_score_10", None)
        grade = res.get("grade", None)

        # Basic presence checks
        required_keys = {"risk_score_10", "grade", "components", "weights", "reasons", "drivers", "context"}
        missing = required_keys - set(res.keys())
        assert not missing, f"[{case['name']}] Missing keys: {missing}"

        # Bounds & types
        assert isinstance(score, (int, float)), f"[{case['name']}] risk_score_10 must be numeric, got {type(score)}"
        assert 0.0 <= float(score) <= 10.0, f"[{case['name']}] risk_score_10 out of bounds: {score}"
        assert grade in allowed_grades, f"[{case['name']}] grade '{grade}' not in {allowed_grades}"

        # Components sanity
        comps = res["components"]
        for k, v in comps.items():
            assert 0.0 <= float(v) <= 1.0, f"[{case['name']}] component {k} out of [0,1]: {v}"

        # Weights sum ≈ 1
        wsum = sum(res["weights"].values())
        assert abs(wsum - 1.0) < 1e-6, f"[{case['name']}] weights must sum to 1.0, got {wsum}"

        # Secured loans should compute an LTV component (present even if 0)
        if case["secured"]:
            assert "ltv" in comps, f"[{case['name']}] secured loans must include 'ltv' component"

        print(f"✅ {case['name']}: score={score}, grade={grade}, reasons={res['reasons']}")

    print("🎉 All risk_agent self-checks passed.")