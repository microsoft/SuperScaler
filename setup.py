from setuptools import setup, find_packages

install_requires = [
    "protobuf==3.8",
    "setuptools>=49.2.0",
    "PyYAML==5.1.2",
    "pytest==5.3.2",
    "humanreadable==0.1.0",
    "bitmath==1.3.3.1"
]
setup(
    name="superscaler",
    version="0.1",
    packages=find_packages(),
    description="SuperScaler Project",
    install_requires=install_requires
)
