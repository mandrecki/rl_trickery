from distutils.core import setup

setup(
    name='RL_trickery',
    version='0.1dev',
    packages=['rl_trickery'],
    author="Marian Andrecki",
    author_email="marian.andrecki@gmail.com",
    description="A bag of tricks and utils for reinforcement learning",
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=[
        'gym'
    ],
    python_requires='>=3.7',
)
