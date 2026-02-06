"""Setup script for robust-vision package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="robust-vision",
    version="1.0.0",
    author="Yahya Akbay",
    author_email="",
    description="Production-Ready Robust Vision: Scalable Training Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/or4k2l/robust-vision",
    project_urls={
        "Bug Tracker": "https://github.com/or4k2l/robust-vision/issues",
        "Source Code": "https://github.com/or4k2l/robust-vision",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "robust-vision-train=scripts.train:main",
            "robust-vision-eval=scripts.eval_robustness:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
