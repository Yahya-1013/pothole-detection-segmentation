from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="pothole-detection-segmentation",
    version="1.0.0",
    author="Yahya-1013",
    description="Pothole Detection using Semantic Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yahya-1013/pothole-detection-segmentation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=required,
)
