# setup.py (at project_root)
from setuptools import setup, find_packages

setup(
    name="modules",
    version="0.1",
    packages=find_packages(include=['exts', 'exts.*', 'scripts', 'scripts.*']),
)