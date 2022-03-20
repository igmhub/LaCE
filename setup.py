#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Lyman-alpha forest flux power spectrum emulator"
version="2.0.1"

setup(name="lace",
    version=version,
    description=description,
    url="https://github.com/igmhub/LaCE",
    author="Chris Pedersen, Andreu Font-Ribera",
    author_email="afont@ifae.es",
    packages=find_packages(),
    )
