from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="hpobench",
    description="Benchmark of hyperparameter optimization algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rick12000/hpo-benchmark",
    author="Riccardo Doyle",
    author_email="r.doyle.edu@gmail.com",
    packages=find_packages(),
    version="0.0.1",
    license="Apache License 2.0",
    install_requires=[line.strip() for line in open("requirements.txt").readlines()],
)
