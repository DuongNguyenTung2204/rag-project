from src.guards.base_guard import BaseGuard
import logging
logger = logging.getLogger(__name__)

class OutputGuard(BaseGuard):
    async def guard(self, response: str, context: str) -> tuple[bool, str]:
        return await self.check_output(response, context)