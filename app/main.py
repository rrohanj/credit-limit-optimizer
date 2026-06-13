from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="Credit Card Limit Optimization Engine",
    description="Production API using traditional ML to evaluate default risk and scale credit lines.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any origin (perfect for testing and local frontends)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)# Load the trained XGBoost model from your serialized checkpoint
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    # Fallback to handle direct execution testing paths
    model = joblib.load("app/model.pkl")

# Define data structures for validation using Pydantic
class ApplicantProfile(BaseModel):
    current_limit: float = Field(..., gt=0, description="Current credit card limit")
    sex: int = Field(..., ge=1, le=2, description="1 = Male, 2 = Female")
    education: int = Field(..., ge=0, le=6, description="Education Tier (1-4 standard tiers)")
    marriage: int = Field(..., ge=0, le=3, description="1 = Married, 2 = Single, 3 = Other")
    age: int = Field(..., gt=18, description="Age of the client")
    repayment_status_last_month: int = Field(..., description="Repayment status (-1=Paid duly, 1=Delay 1 month, 2=Delay 2 months)")
    current_bill_balance: float = Field(..., description="Most recent statement balance (BILL_AMT1)")
    previous_bill_balance: float = Field(..., description="Statement balance from the prior month (BILL_AMT2)")
    amount_paid_last_month: float = Field(..., ge=0, description="Amount paid in the last cycle (PAY_AMT1)")

@app.get("/")
def health_check():
    return {"status": "healthy", "engine": "XGBoost Tabular Risk Engine V1.0"}

@app.post("/optimize-limit")
def optimize_credit_limit(profile: ApplicantProfile):
    # 1. Pipeline Feature Engineering Match
    # Calculate Utilization Rate matching notebook clip boundaries [0.0, 1.5]
    utilization_rate = profile.current_bill_balance / (profile.current_limit + 1e-5)
    utilization_rate = float(np.clip(utilization_rate, 0.0, 1.5))
    
    # Calculate Payment-to-Minimum Ratio matching notebook clip boundaries [0.0, 20.0]
    expected_min_due = profile.previous_bill_balance * 0.05
    if expected_min_due > 0:
        pay_to_min_ratio = profile.amount_paid_last_month / expected_min_due
    else:
        pay_to_min_ratio = 1.0
    pay_to_min_ratio = float(np.clip(pay_to_min_ratio, 0.0, 20.0))
    
    # 2. Vector Matrix Compilation
    input_vector = np.array([[
        profile.current_limit,
        profile.sex,
        profile.education,
        profile.marriage,
        profile.age,
        profile.repayment_status_last_month,
        utilization_rate,
        pay_to_min_ratio
    ]])
    
    # 3. Model Inference Extraction
    probability_of_default = float(model.predict_proba(input_vector)[0][1])
    
    # 4. Bank Risk Portfolio Optimization Strategy
    current_limit = profile.current_limit
    
    if probability_of_default < 0.15:
        # Tier 1 Elite Client: Strong 30% limit increment approval
        decision = "Approve Aggressive Credit Extension"
        optimized_limit = current_limit * 1.30
    elif probability_of_default < 0.35:
        # Tier 2 Steady Client: Low-risk conservative 10% step increment
        decision = "Approve Standard Credit Extension"
        optimized_limit = current_limit * 1.10
    elif probability_of_default < 0.60:
        # Tier 3 High Exposure Client: Maintain credit ceiling freeze
        decision = "Maintain Existing Limit Freeze"
        optimized_limit = current_limit
    else:
        # Tier 4 Default Risk Threat: Proactively mitigate credit line risk by 20%
        decision = "Risk Mitigation Protocol Triggered: Drawdown Limit"
        optimized_limit = current_limit * 0.80

    # Clean rounding to standard currency denominations
    optimized_limit = round(optimized_limit, -3)

    return {
        "features_calculated": {
            "utilization_rate": round(utilization_rate, 4),
            "payment_to_minimum_coverage": round(pay_to_min_ratio, 4)
        },
        "risk_evaluation": {
            "predicted_probability_of_default": round(probability_of_default, 4),
            "risk_status": "Low" if probability_of_default < 0.25 else "Medium" if probability_of_default < 0.55 else "High"
        },
        "portfolio_action": {
            "strategy": decision,
            "baseline_limit": current_limit,
            "recommended_optimized_limit": optimized_limit
        }
    }