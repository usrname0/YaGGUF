"""
Shared theme configuration for terminal output colors.
"""

from colorama import Fore, Style

# Theme for terminal colors
THEME = {
    "info": Fore.WHITE + Style.DIM,
    "success": Fore.GREEN,
    "warning": Fore.YELLOW,
    "error": Fore.RED,
    "highlight": Fore.CYAN,
    "metadata": Fore.WHITE + Style.DIM,
}
