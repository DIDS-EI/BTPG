from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='BTPG',
    version='0.1.0',
    packages=['btgym'],
    install_requires=required,
    author='DIDSL-EI',
    author_email='',
    description=' Platform and Benchmark for Behavior TreePlanning in Everyday Service Robots',
    url='https://github.com/DIDSL-EI/BTPG',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

