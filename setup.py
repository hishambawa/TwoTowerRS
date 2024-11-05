from setuptools import setup

setup(
    name='ttrs',
    version='1.0.0',
    packages=['ttrs'],
    package_dir={'':'src'},
    install_requires=[
        'tf-keras==2.15.0',
        'tensorflow==2.15.0',
        'tensorflow_recommenders==0.7.2',
        'structlog==24.4.0'
    ],
)