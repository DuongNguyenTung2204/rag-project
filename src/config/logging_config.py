import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from config.settings import settings
    LOG_LEVEL = getattr(logging, settings.log_level.upper(), logging.INFO)
    LOGS_DIR = Path("src") / getattr(settings.paths, "logs_dir", "logs")
except (ImportError, AttributeError):
    LOG_LEVEL = logging.INFO
    LOGS_DIR = Path("src") / "logs"

def setup_logging(
    console_level: Optional[int] = None,
    file_level: int = logging.DEBUG,
    override_root_level: bool = True
) -> Path:
    """
    Thiết lập logging toàn cục một cách an toàn và đầy đủ.

    Args:
        console_level: Level cho console (mặc định theo settings hoặc INFO)
        file_level: Level cho file log (mặc định DEBUG để lưu chi tiết)
        override_root_level: Buộc root logger xuống mức thấp nhất cần thiết

    Returns:
        Path: đường dẫn file log vừa tạo
    """
    effective_console_level = console_level or LOG_LEVEL

    # ───────────────────────────────────────────────
    # Bước 1: Buộc root logger chấp nhận mức thấp nhất cần thiết
    # ───────────────────────────────────────────────
    root = logging.getLogger()
    
    # Để file có thể nhận DEBUG → root phải ≤ DEBUG
    min_level_needed = min(effective_console_level, file_level)
    if override_root_level:
        root.setLevel(min_level_needed)
    
    # Xóa hết handler cũ (rất quan trọng khi reload)
    root.handlers.clear()

    # ───────────────────────────────────────────────
    # Bước 2: Formatter thống nhất
    # ───────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ───────────────────────────────────────────────
    # Bước 3: Console handler
    # ───────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(effective_console_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ───────────────────────────────────────────────
    # Bước 4: File handler - luôn chi tiết
    # ───────────────────────────────────────────────
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"rag_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # ───────────────────────────────────────────────
    # Bước 5: Tắt / giảm log rác từ các thư viện phổ biến
    # ───────────────────────────────────────────────
    noisy_loggers = [
        "watchfiles",           # Chainlit watch mode
        "chainlit",
        "chromadb",
        "chromadb.telemetry",
        "httpx",                # HTTP client
        "httpcore",
        "asyncio",
        "sentence_transformers",
        "llama_index",
        "openai",               # Nếu dùng openai client
        "groq",                 # Groq client
        "redis",                # Redis client
        "urllib3",
        "pymongo",              # Root pymongo
        "pymongo.command",      # Command started/succeeded (rất noisy)
        "pymongo.connection",   # Connection checkout/checkin
        "pymongo.serverSelection",  # Server selection
        "pymongo.topology",     # Topology changes
        "pymongo.pool",         # Connection pool
    ]

    for lg_name in noisy_loggers:
        lg = logging.getLogger(lg_name)
        lg.setLevel(logging.ERROR)
        lg.propagate = False
        # Cách mạnh hơn nếu cần tắt hoàn toàn:
        # lg.disabled = True
        # lg.setLevel(logging.CRITICAL + 5)

    # Tắt watchfiles triệt để hơn
    watch = logging.getLogger("watchfiles")
    watch.setLevel(100)           # Cao hơn CRITICAL
    watch.disabled = True
    watch.propagate = False

    # ───────────────────────────────────────────────
    # Bước 6: Xác nhận setup
    # ───────────────────────────────────────────────
    root.info(
        "Logging đã được thiết lập thành công\n"
        f"  • Root level     : {logging.getLevelName(root.level)}\n"
        f"  • Console level  : {logging.getLevelName(console.level)}\n"
        f"  • File level     : {logging.getLevelName(file_handler.level)}\n"
        f"  • File lưu tại   : {log_file}"
    )

    # Test DEBUG ngay lập tức → dễ kiểm tra
    root.debug("Logging DEBUG đã hoạt động (kiểm tra trong file log)")

    return log_file