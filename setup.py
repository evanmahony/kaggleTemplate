from distutils.core import setup

setup(
    name="KaggleTemplate",
    version="0.1.0",
    author="Evan Mahony",
    author_email="evan99mahony@gmail.com",
    packages=["src.example"],
    scripts=["src/main.py"],
    url="https://github.com/evanmahony/kaggleTemplate",
    license="LICENSE",
    description="A template for starting Kaggle competitions with PyTorch.",
    long_description=open("README.md").read(),
    install_requires=["torch", "tensorboard"],
)
