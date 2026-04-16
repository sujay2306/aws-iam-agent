from setuptools import setup

setup(
    name="aws-iam-agent",
    version="2.0.0",
    description="AI-powered AWS IAM agent",
    py_modules=["main"],
    install_requires=[
        "langchain",
        "langchain-openai",
        "boto3",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "aws-iam-agent=main:main",
        ],
    },
)
