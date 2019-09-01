from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='AMLP',
    version='0.1.0',
    description='Automated Machine Learning Pipeline',
    long_description=readme,
    author='Yonglin Zhu',
    author_email='zhuygln@gmail.com',
    url='https://github.com/zhuygln/amlp',
    license=MIT
    )
