from typing import Annotated
from fastapi import Depends

from app.configs.config import Settings, get_settings
from app.services.rag_service import RagService


SettingsDep = Annotated[Settings, Depends(get_settings)]


class RagServiceSingleton:
    _instance = None
    _settings = None

    @classmethod
    def get_instance(cls, settings: SettingsDep):
        if cls._instance is None or cls._settings != settings:
            cls._instance = RagService(settings)
            cls._settings = settings
        return cls._instance


def get_rag_service(settings: SettingsDep):
    return RagServiceSingleton.get_instance(settings)


RagDep = Annotated[RagService, Depends(get_rag_service)]
