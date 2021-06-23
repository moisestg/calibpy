from setuptools import find_packages, setup

with open('requirements.txt') as fin:
    install_required = fin.read().splitlines()

with open('test-requirements.txt') as fin:
    test_required = fin.read().splitlines()

setup(
    name='calibpy',
    packages=find_packages(include=['calibpy']),
    version='0.1.0',
    description='Tiny camera calibration library',
    author='moisestg',
    license='MIT',
    install_requires=install_required,
    setup_requires=['pytest-runner>=2.0,<3dev'],
    tests_require=test_required,
    test_suite='tests',
)
