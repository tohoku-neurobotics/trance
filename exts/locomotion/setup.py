"""Installation script for the 'locomotion' python package."""

from setuptools import setup, find_packages
import os
import toml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
EXTENSION_TOML = os.path.join(PROJECT_ROOT, "config", "extension.toml")

EXTENSION_TOML_DATA = toml.load(EXTENSION_TOML)

setup(
    name="locomotion",
    packages=find_packages(where="."),  # looks inside exts/locomotion/
    package_dir={"": "."},              # root is current folder
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=["psutil"],
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
