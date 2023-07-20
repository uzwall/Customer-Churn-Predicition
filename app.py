from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


app = FastAPI()


@app.get("/home")
async def root():
    print( {'response':"200","message": "Hello World"})
    return {'response':"200","message": "Hello World"}

if __name__=="__main__":
    uvicorn.run("app:app",host="localhost",port=8080,reload=True)


class Employee(BaseModel):
    name: str
    salary: float=10000.0
    age: int


@app.post("/employee")
async def insert(emp:Employee):
    print(emp)
    return {"response":"200","message":"inserted successfully"}





