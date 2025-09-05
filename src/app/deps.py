import logging
from typing import Annotated
from fastapi import Depends

from app.configs.config import Settings, get_settings
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

SettingsDep = Annotated[Settings, Depends(get_settings)]


class RagServiceSingleton:
    _instance: RagService | None = None
    _initialized: bool = False

    @classmethod
    def get_instance(cls, settings: Settings) -> RagService | None:
        if not cls._initialized:
            cls._instance = RagService(settings)
            cls._initialized = True

            logger.info("RagService singleton initialized")

        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None
        cls._initialized = False


def get_rag_service(settings: SettingsDep) -> RagService | None:
    return RagServiceSingleton.get_instance(settings)


RagDep = Annotated[RagService, Depends(get_rag_service)]
