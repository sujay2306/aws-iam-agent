from setuptools import setup

setup(
    name="aws-iam-agent",
    version="1.0.0",
    py_modules=["main"],          # ← flat, no package folder needed
    install_requires=[
        "langchain",
        "langchain-openai",
        "boto3",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "aws-iam-agent=main:main",   # ← points to main.py → main()
        ],
    },
)
