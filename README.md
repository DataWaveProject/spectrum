## spectrum
A command-line tool for computing power spectrum from spherical harmonic coefficients in a netCDF dataset.
The spherical harmonic coefficients must be consistent with CDO and SHTns indexing.

### Usage:
spectrum [-h] [-o OUTPUT] [-t TRUNCATION] [-select VARIABLES] files [files ...]

### Options:
  -h, --help            show this help message and exit 
  -o OUTPUT, --output OUTPUT
                        output file name
  -t TRUNCATION, --truncation TRUNCATION
                        triangular truncation
  -select VARIABLES, --variables VARIABLES
                        compute spectra for selected variables
