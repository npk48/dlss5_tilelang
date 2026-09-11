"""Small PTX, packed-Half and memory instruction adapters; no CUDA kernel bodies."""
from pathlib import Path


def instruction_source(name):
    return Path(__file__).with_name(name).read_text()
