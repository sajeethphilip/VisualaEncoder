from setuptools import setup, find_packages

setup(
    name="autoencoder-tool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "streamlit",
        "opencv-python",
        "pillow",
        "pytest",
    ],
    python_requires=">=3.8",
)
