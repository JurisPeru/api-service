import logging
from fastapi import APIRouter

router = APIRouter(prefix="/health")

logger = logging.getLogger(__name__)


@router.get("/", include_in_schema=False)
async def get_health():
    logger.info("status ok")
    return {"status": "ok"}
