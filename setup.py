from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="trading_rl_agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=Path("requirements.txt").read_text().splitlines(),
    extras_require={
        "dashboard": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
            "websockets>=11.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-rl-agent=main:app",
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
