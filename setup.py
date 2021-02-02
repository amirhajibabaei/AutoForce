# +
from distutils.core import setup


setup(
    name="theforce",
    version="v2021.01",
    author="Amir HajiBabaei T.",
    author_email="a.hajibabaei.86@gmail.com",
    description="machine learning of the potential energy surface from DFT",
    url="https://github.com",
    package_dir={'theforce': 'theforce'},
    packages=['theforce'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

# new
# theforce.cl.relax
# theforce.cl.init_model
# theforce.calculator.meta -> Qlvar
