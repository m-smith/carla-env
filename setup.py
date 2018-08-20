from setuptools import setup

setup(
    name='carla-env',
    version='0.1',
    install_requires=["gym", "numpy"],
    description='Gym API CARLA environments for Reinforcement Learning',
    author='Matthew J. A. Smith',
    packages=['carla_env']
)
