from setuptools import setup, find_packages

setup(
    name="trading_rl_agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # dependencies will be installed from requirements.txt
    ],
)
