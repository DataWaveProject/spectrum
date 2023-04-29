import os
import shutil

from numpy.distutils.core import setup, Extension

# Clean build directory to force recompilation
build_dir = 'build'
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)

setup(
    name='spectrum',
    version='0.1',
    license='GPLv3',
    description='A Python package for computing power spectra '
                'from CDO spherical harmonic coefficients.',
    author='Yanmichel A. Morfa',
    author_email='yanmichel.morfa-avalos@mpimet.mpg.de',
    packages=['spectrum'],
    ext_modules=[Extension('numeric_tools',
                           sources=['fortran_source/numeric_tools.f90'])
                 ],
    entry_points={
        'console_scripts': [
            'spectrum=spectrum.compute_spectra:main'
        ]
    },
    setup_requires=['numpy'],
    install_requires=[
        'numpy>=1.20.0',
        'xarray',
        'pint>=0.20.1',
    ],
    include_package_data=True,
    package_data={'spectrum': ['*.so']},
    zip_safe=False
)
