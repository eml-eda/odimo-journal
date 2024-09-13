import setuptools

setuptools.setup(
    name='odimo',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=['setuptools'],
    maintainer='Matteo Risso',
    maintainer_email='matteo.risso@polito.it',
    description='ODiMO',
    license='Apache License 2.0',
    tests_require=['unittest'],
    python_requires=">=3.9",
)
