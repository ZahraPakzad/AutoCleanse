from setuptools import setup

setup(
    name="AutoEncoder",
    packages=["AutoEncoder"],
    version="1.0.0",
    author="Tung Tran",
    description="Data cleaning and anonymization using autoencoder",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JohanObluda/AutoEncoder",
    license="MIT",
    include_package_data=True,
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
