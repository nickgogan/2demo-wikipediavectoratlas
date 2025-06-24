from setuptools import setup, find_packages

setup(
    name="wikipedia-vector",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "sentence-transformers>=2.2.2",
        "pymongo>=4.5.0,<5.0.0",
        "python-dotenv>=1.0.0",
        "datasets>=2.14.0",
        "pympler>=1.0.1",
        "hurry.filesize>=0.9"
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "flake8-bugbear>=23.0.0",
            "pre-commit>=3.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "wikipedia-vector=wikipedia_vector.main:main"
        ]
    }
)
