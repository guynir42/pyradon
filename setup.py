from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyradon",
    version="0.1.0",
    author="Guy Nir",
    author_email="guy.nir@weizmann.ac.il",
    description="Streak detection in astronomical images using the Fast Radon Transform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guynir42/pyradon",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.2",
        "scipy>=1.10.1",
        "matplotlib>=3.7.0",
        "h5py>=3.8.0",
    ],
    extras_require={
        "test": ["pytest", "flaky"],
    },
)