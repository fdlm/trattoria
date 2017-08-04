from setuptools import setup

setup(
    name='Trattoria',
    version='0.1dev',
    description='Training Tools for Lasagne',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license='MIT',
    author='Filip Korzeniowski',
    author_email='filip.korzeniowski@jku.at',
    install_requires=['numpy', 'Lasagne', 'Theano', 'tqdm', 'PyYAML']
)
