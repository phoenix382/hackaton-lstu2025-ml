from fastapi import FastAPI, Request, HTTPException
from config import app, gen_first_plan, upd_plan

@app.post("/ml/get_first_plan")
async def generate_first_plan(request: Request):
    input_data = await request.json()
    plan = gen_first_plan(input_data)
    return plan

@app.post("/ml/update_plan")
async def update_plan(request: Request):
    input_data = await request.json()
    result = upd_plan(input_data)
    return result