"""screening-core — shared LLM evaluation infrastructure for screening pipeline apps."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("screening-core")
except PackageNotFoundError:
    __version__ = "0.0.0"
