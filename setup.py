from setuptools import setup, find_packages

setup(
    name="llm_thought_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.4.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=4.1.0",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for analyzing reasoning processes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
