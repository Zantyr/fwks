import os
import setuptools

descr = open("README.md").read()

setuptools.setup(
    name="fwks",
    version="0.1",
    author="Paweł Tomasik",
    author_email="tomasik.kwidzyn@gmail.com",
    description="A package for declarative specification of speech processing pipelines",
    long_description=descr,
    long_description_content_type="text/markdown",
    url="https://github.com/Zantyr/ASR",
    packages=["fwks"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    package_data={
        'fwks': [os.path.join('etc', '*')],
    },
    install_requires=[
        "dill",
        "keras",
        "numpy",
        "scipy",
        "sklearn",
        "syntax",
        "tqdm",
        # pynini is recommended
        # "git+https://github.com/detly/gammatone.git"
        "librosa",
        "luigi"
    ]
)
