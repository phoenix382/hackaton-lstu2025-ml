from fastapi import FastAPI

app = FastAPI()

@app.get("/ml/hello")
async def read_root():
    return {"message": "hello world"}
