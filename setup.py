from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='doccano-transformer',
    version='1.0.0',
    description='Format transformer tool for doccano',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/doccano/doccano-transformer',
    author='Hiroki Nakayama, Yasufumi Taniguchi',
    author_email='hiroki.nakayama.py@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='doccano,annotation,machine learning',
    packages=find_packages(where='doccano_transformer'),
    python_requires='>=3.5, <4',
    install_requires=['spacy', 'importlib-metadata'],
)
