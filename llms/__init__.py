import importlib
import pkgutil
from pathlib import Path

from .base import BaseLLM, LLMFactory

def _import_all_providers():
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name not in ['base', '__init__']:
            try:
                importlib.import_module(f'.{module_info.name}', package=__package__)
            except ImportError as e:
                # Skip providers with missing dependencies
                pass

_import_all_providers()

__all__ = ['BaseLLM', 'LLMFactory'] 