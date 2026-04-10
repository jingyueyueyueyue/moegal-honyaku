from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.model_sync import ensure_models_ready
from app.services.ocr import warmup_models


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_models_ready()
    warmup_models()
    yield


def create_app() -> FastAPI:
    from app.api.routes import register_routers
    from app.core.env_sync import sync_env_files
    
    # 启动时同步 .env.example 到 .env
    sync_env_files()

    application = FastAPI(lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routers(application)
    return application


app = create_app()
