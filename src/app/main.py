from fastapi import FastAPI
from app.api.routes import health, ask
from fastapi.middleware.cors import CORSMiddleware

from app.configs.config import setup_logging


setup_logging()
app = FastAPI(title="API Law Services")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router=health.router)
app.include_router(router=ask.router, prefix="/api")
