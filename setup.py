from pathlib import Path
from setuptools import setup, find_packages

# Read dependencies from requirements.txt
req_file = Path(__file__).parent / "requirements.txt"
requirements = []
if req_file.exists():
    with open(req_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.strip().startswith("#")
        ]

setup(
    name="Sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
)