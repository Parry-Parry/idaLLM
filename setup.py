import setuptools

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='idallm',
    version='0.0.1',
    author='Andrew Parry',
    author_email='a.parry.1@research.gla.ac.uk',
    description="IDA Cluster LLM API Utility",
    url='https://github.com/Parry-Parry/idaLLM',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
)
