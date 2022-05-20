from setuptools import setup, find_packages
import os
import re


def read_version():
    # importing gpustat causes an ImportError :-)
    __PATH__ = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(__PATH__, 'nomeroff_net/__init__.py')) as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find __version__ string")


# get project version
__version__ = read_version()

# get project long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# get project requirements list
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()
    required_pkgs, required_repos = requirements.split('# git repos')
    required_pkgs = required_pkgs.split()
    required_repos = required_repos.split()

setup(name='nomeroff-net',
      version=__version__,
      description='Automatic numberplate recognition system',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='ai nomeroffnet yolov5 craft ocr rnn opensource license number plate recognition '
               'licenseplate numberplate license-plate number-plate ria-com ria.com ria',
      url='https://github.com/ria-com/nomeroff-net',
      author='Dmytro Probachay, Oleg Cherniy',
      author_email='dimabendera@gmail.com, oleg.cherniy@ria.com',
      license='GNU General Public License v3.0',
      packages=find_packages(),
      install_requires=required_pkgs,
      dependency_links=required_repos,
      include_package_data=True,
      python_requires='>=3.7',
      zip_safe=False)
