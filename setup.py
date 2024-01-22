from setuptools import setup

setup(
    name="AutoCleanse",
    packages=["AutoCleanse"],
    version="1.0.3",
    author="Tung Tran",
    description="A tool for data cleaning and anonymization using autoencoder",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tungsontran/AutoCleanse",
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
