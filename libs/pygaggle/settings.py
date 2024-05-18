from pathlib import Path
import os

from pydantic import BaseSettings


class Settings(BaseSettings):
    # Cache settings
    cache_dir: Path = Path(os.getenv('XDG_CACHE_HOME',
                                     str(Path.home() / '.cache'))) / 'ranking'
    flush_cache: bool = False
