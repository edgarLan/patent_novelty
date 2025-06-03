from setuptools import setup, find_packages

setup(
    name='patent_novelty',
    version='0.1.0',
    description='Tools for detecting novelty, surprise, and divergence in patent text data.',
    author='Ton Nom',
    author_email='ton.email@example.com',
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    install_requires=[
        'pandas>=1.3',
        'numpy>=1.21',
        'scipy>=1.7',
        'nltk>=3.6',
        'spacy==3.7.2',
        'tqdm>=4.60',
        'scikit-learn>=1.0',
        'statsmodels>=0.13',
        'rbo>=0.1.3',
        'joblib>=1.2',
        'fast-cdindex @ git+https://github.com/dspinellis/fast-cdindex.git',
        'openpyxl'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.8',
)
