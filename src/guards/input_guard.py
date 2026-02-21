from src.guards.base_guard import BaseGuard
import logging
logger = logging.getLogger(__name__)

class InputGuard(BaseGuard):
    async def guard(self, query: str) -> tuple[bool, str]:
        return await self.check_input(query)