from setuptools import setup, find_packages

setup(
    name="genai-medical-imaging",
    version="1.0.0",
    author="Vivek Nagappa",
    author_email="vivekjalakote@gmail.com",
    description="Generative AI for Medical Imaging & Drug Discovery",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ]
)
