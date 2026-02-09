from setuptools import setup, find_packages

setup(
    name="sam-annotator",
    version="0.1.0",
    description="Interactive annotation tool using Segment Anything (SAM)",
    author="Orlando Reyna",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch",
        "opencv-python",
        "dataclasses-json",
        "jupyter-bbox-widget",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git"
    ],
)