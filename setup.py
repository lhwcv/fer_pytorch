import setuptools, os
from setuptools import  find_packages

PACKAGE_NAME = 'fer_pytorch'
VERSION = '0.1.0'
AUTHOR = 'lhwcv'
EMAIL = 'liahoweicv@hotmail.com'
DESCRIPTION = 'Face Expression Recognition in Pytorch'
GITHUB_URL = 'https://github.com/lhwcv/fer_pytorch'


setuptools.setup(
    name = PACKAGE_NAME,
    version = VERSION,
    author = AUTHOR,
    author_email = EMAIL,
    description = DESCRIPTION,
    long_description="",
    long_description_content_type='text/markdown',
    url = GITHUB_URL,
    packages=find_packages(exclude=('weights','configs')),
    package_dir={'fer_pytorch':'./fer_pytorch'},
    package_data={'fer_pytorch.face_detect': ['data/*net.pt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'opencv-python'
    ],
)
