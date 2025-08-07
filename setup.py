from setuptools import setup, find_packages

setup(
    name='city_learning',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'xgboost',
        'lightgbm'
    ],
    entry_points={
        'console_scripts': [
            'city-train=src.train:main',
            'city-eval=src.evaluate:main'
        ],
    },
)