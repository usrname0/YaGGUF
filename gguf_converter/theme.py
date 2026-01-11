"""
Shared theme configuration for terminal output colors.
"""

from typing import Dict
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform color support
colorama_init(autoreset=True)

# Theme for terminal colors
THEME: Dict[str, str] = {
    "info": Fore.WHITE + Style.DIM,
    "success": Fore.GREEN,
    "warning": Fore.YELLOW,
    "error": Fore.RED,
    "highlight": Fore.CYAN,
    "metadata": Fore.WHITE + Style.DIM,
}
