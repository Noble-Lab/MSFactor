#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()

test_requirements = [
    'pytest', 'coverage', "flake8"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ms_imputer',
    version='0.1.0',
    description="Impute missing values using NMF",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Lincoln Harris",
    author_email='lincolnh@uw.edu',
    url='https://github.com/lincoln-harris/ms_imputer',
    packages=[
        'ms_imputer',
    ],
    package_dir={'ms_imputer':
                     'ms_imputer'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='ms_imputer',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={
        'console_scripts': [
            'cerebra = cerebra.commandline:cli'
        ]
    },
    test_suite='tests',
    tests_require=test_requirements
)