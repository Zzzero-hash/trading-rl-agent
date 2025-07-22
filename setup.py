from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="trade-agent",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=Path("requirements.txt").read_text().splitlines(),
    entry_points={
        "console_scripts": [
            "trade-agent=trade_agent.cli:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
