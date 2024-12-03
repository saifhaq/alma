from setuptools import setup

try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup

if __name__ == "__main__":
    setup(
        name="alma",
        version="0.0.1",
        packages=find_packages(),
    )
