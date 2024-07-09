from fastapi import FastAPI

from src.routers import predict_router

app = FastAPI()

app.include_router(predict_router.router)
# app.include_router(prediction.router)