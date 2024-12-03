from setuptools import setup

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

if __name__ == "__main__":
    setup(
        name="alma",
        version="0.0.1",
        packages=find_packages(),
    )
