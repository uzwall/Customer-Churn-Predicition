from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib

backend.app = FastAPI()

# Load the model
model = joblib.load('model\prediction2.joblib')

    
class Customer(BaseModel):
    AccountWeeks: float
    ContractRenewal: int
    DataPlan: int
    DataUsage: float
    CustServCalls: int
    DayMins: float
    DayCalls: int
    MonthlyCharge: float
    OverageFee: float
    RoamMins: float


@backend.app.get("/home")
def home():
    print("hello")
    return {"message": "Hello World"}


#function to print the churn prediction using trained model
@backend.app.post("/infer")
def infer(data:Customer):
    
    # Convert the data to a 2D array
    input_data = [[
        data.AccountWeeks, data.ContractRenewal, data.DataPlan,
        data.DataUsage, data.CustServCalls, data.DayMins,
        data.DayCalls, data.MonthlyCharge, data.OverageFee,
        data.RoamMins
    ]]
    
    # Make predictions using the model
    predictions = model.predict(input_data)
    
    if predictions[0]==1:
        return {"Prediction":"Customer is more likely to churn"}
    else:
        return {"Predictions":"Customer will not churn"}


if __name__=="__main__":
    import backend.app as app
    uvicorn.run("app:app",host="0.0.0.0", port=8084, reload=True)