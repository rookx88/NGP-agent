from setuptools import setup, find_packages

setup(
    name="ngp_agent",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'requests',
        'numpy',
        'python-dotenv',
        'sentence-transformers',
        'openai'
    ]
) 