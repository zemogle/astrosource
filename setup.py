from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.pip'), encoding='utf-8') as f:
    requirements=[line.strip() for line in f.readlines()]

setup(
    name='astrosource',
    version='1.1.2',
    description='Analysis script for sources with variability in their brightness',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Michael Fitzgerald and Edward Gomez',
    url="https://github.com/zemogle/astrosource",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'astrosource=astrosource.main:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
