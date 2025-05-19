from setuptools import setup, find_packages

setup(
    name="trading_rl_agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={
        'console_scripts': [
            'trade-agent=trading_rl_agent.main:main',
        ],
    },
)
