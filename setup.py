__author__ = "Maxim Ziatdinov"
__copyright__ = "Copyright Maxim Ziatdinov (2021)"
__maintainer__ = "Maxim Ziatdinov"
__email__ = "maxim.ziatdinov@ai4microcopy.com"

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(module_dir, 'gpax/__version__.py')) as f:
    __version__ = f.read().split("'")[1]

if __name__ == "__main__":
    setup(
        name='gpax',
        python_requires='>=3.7',
        version=__version__,
        description='Gaussian processes in NumPyro and JAX',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        long_description_content_type='text/markdown',
        url='https://github.com/ziatdinovmax/gpax/',
        author='Maxim Ziatdinov',
        author_email='maxim.ziatdinov@ai4microcopy.com',
        license='MIT license',
        packages=find_packages(),
        zip_safe=False,
        install_requires=[
            'jax>=0.2.21',
            'numpyro>=0.8.0',
            'dm-haiku>=0.0.5',
            'matplotlib>=3.1'
        ],
        classifiers=['Programming Language :: Python',
                     'Development Status :: 3 - Alpha',
                     'Intended Audience :: Science/Research',
                     'Operating System :: POSIX :: Linux',
                     'Operating System :: MacOS :: MacOS X',
                     'Topic :: Scientific/Engineering']
    )
