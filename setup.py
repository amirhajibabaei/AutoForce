# +
from distutils.core import setup


setup(
    name="theforce",
    version="v2021.05",
    author="Amir Hajibabaei",
    author_email="a.hajibabaei.86@gmail.com",
    description="machine learning of the ab initio potential energy surface",
    url="https://github.com",
    package_dir={'theforce': 'theforce'},
    packages=['theforce'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
