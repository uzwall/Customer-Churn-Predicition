import gradio as gr
import joblib
import requests

# Function that takes the input_data and returns the result
def predict_churn(AccountWeeks, ContractRenewal, DataPlan, DataUsage, CustServCalls,
                  DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins):
    # Assuming you have a function or model that predicts churn
    input_data = [[AccountWeeks, ContractRenewal, DataPlan, DataUsage, CustServCalls,
                   DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins]]
    result = requests.posts("http://localhost:8084/infer", json=input_data)
    return "Customer is more likely to churn" if result[0] == 1 else "Customer will not churn"

# Gradio Inputs
inputs = [
    gr.inputs.Number(label="AccountWeeks"),
    gr.inputs.Number(label="ContractRenewal"),
    gr.inputs.Number(label="DataPlan"),
    gr.inputs.Number(label="DataUsage"),
    gr.inputs.Number(label="CustServCalls"),
    gr.inputs.Number(label="DayMins"),
    gr.inputs.Number(label="DayCalls"),
    gr.inputs.Number(label="MonthlyCharge"),
    gr.inputs.Number(label="OverageFee"),
    gr.inputs.Number(label="RoamMins")
]

# Gradio Output
output = gr.outputs.Label()  # Assuming the output is a label showing the churn prediction result

# Create the Gradio Interface
interface = gr.Interface(fn=predict_churn, inputs=inputs, outputs=output)



# If this script is run as the main module, start the Gradio app
if __name__ == "__main__":
    interface.launch(share=True,server_name=0.0.0.0,server_port=8084)
