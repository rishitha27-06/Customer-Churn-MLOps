import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint(client):
    customer = {
        "data": {
            "Gender": "Male",
            "Senior_Citizen": "No",
            "Tenure_Months": 12,
            "Contract_Type": "Month-to-month",
            "Internet_Service": "DSL",
            "Payment_Method": "Electronic check",
            "Monthly_Charges_INR": 800,
            "Total_Charges_INR": 9600,
            "Tech_Support": "No",
            "Online_Security": "No",
            "Support_Tickets_6M": 5,
            "Avg_Data_Usage_GB": 10,
            "Tenure_Category": "0-12 Months",
            "Avg_Monthly_Spending_INR": 800,
            "Has_Tech_Support": 0,
            "Has_Online_Security": 0,
            "Has_Internet": 1
        }
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200
    assert response.json()["prediction"] in ["Churn", "Not Churn"]
    assert 0 <= response.json()["churn_probability"] <= 1