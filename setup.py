#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

setup(
    name='patchldmseg',
    version='1.0.0',
    author='Nico Loesch',
    author_email='nico.loesch@student.uts.edu.au',
    description='Official Pytorch implementation of the paper "Three-dimensional latent diffusion model for brain tumour segmentation"',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    packages=find_packages(),
    license='private',
    keywords='None',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only'
    ],
    install_requires=[
        'torch',
        'torchvision',
        'pytorch-lightning[extra]>=2.0',
        'torchio>=0.18.90',
        'nibabel',
        'numpy',
        'matplotlib',
        'pandas',
        'torchmetrics[image]',
        'tensorboard',
        'SimpleITK',
        'tqdm',
        'wandb',
        'Pillow',
        'ema-pytorch',
        'scikit-image',
        'ddt',
    ],
    python_requires='>3.10',
)
