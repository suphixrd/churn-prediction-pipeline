import requests
import json
import random

API_URL = "http://127.0.0.1:8000"

# Sample customer data
sample_customers = [
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 840.0
    },
    {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 48,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.20,
        "TotalCharges": 2650.0
    }
]

def test_prediction():
    print(" Testing prediction endpoint...")
    
    for i, customer in enumerate(sample_customers):
        response = requests.post(f"{API_URL}/predict", json=customer)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n Customer {i+1} Prediction:")
            print(f"  Churn: {'Yes' if result['churn_prediction'] == 1 else 'No'}")
            print(f"  Probability: {result['churn_probability']:.2%}")
            print(f"  Risk Level: {result['risk_level']}")
        else:
            print(f" Error: {response.status_code}")

def test_stats():
    print("\n Getting prediction statistics...")
    response = requests.get(f"{API_URL}/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print(json.dumps(stats, indent=2))
    else:
        print(f" Error: {response.status_code}")

def simulate_traffic(num_requests=50):
    print(f"\n Simulating {num_requests} API requests...")
    
    for i in range(num_requests):
        # Randomly select a customer template
        customer = random.choice(sample_customers).copy()
        
        # Add some randomization
        customer['tenure'] = random.randint(1, 72)
        customer['MonthlyCharges'] = round(random.uniform(20, 120), 2)
        customer['TotalCharges'] = round(customer['tenure'] * customer['MonthlyCharges'], 2)
        
        response = requests.post(f"{API_URL}/predict", json=customer)
        
        if response.status_code == 200:
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")
        else:
            print(f" Request {i + 1} failed")
    
    print(" Simulation completed!")

if __name__ == "__main__":
    print("="*50)
    print("CHURN PREDICTION API TESTER")
    print("="*50)
    
    # Test basic predictions
    test_prediction()
    
    # Simulate traffic
    simulate_traffic(100)
    
    # Get stats
    test_stats()