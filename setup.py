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
    author_email="yahya.akbay@example.com",
    description="Production-Ready Robust Vision: Scalable Training Framework with Advanced Techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/or4k2l/robust-vision",
    project_urls={
        "Bug Tracker": "https://github.com/or4k2l/robust-vision/issues",
        "Source Code": "https://github.com/or4k2l/robust-vision",
        "Documentation": "https://github.com/or4k2l/robust-vision/tree/main/docs",
        "Changelog": "https://github.com/or4k2l/robust-vision/releases",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: JAX",
    ],
    keywords="deep-learning computer-vision robustness jax flax training machine-learning",
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "coverage>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "pre-commit>=3.5.0",
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
