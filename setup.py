import setuptools

required = []

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AutoEncoder",
    version="0.1.0",
    author="Tung Tran",
    description="Data cleaning and anonymization using autoencoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohanObluda/AutoEncoder",
    packages=setuptools.find_packages(),
    license="MIT",
    include_package_data=True,
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)