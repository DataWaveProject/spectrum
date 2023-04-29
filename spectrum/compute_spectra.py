"""
    **Description**

    A collection of functions to compute the cross-power spectrum of two sets of spherical
    harmonics coefficients clm1 and clm2. Total cross-power is defined as the integral of the
    clm1 times the conjugate of clm2 over all space, divided by the area the functions span over
    all angular orders as a function of spherical harmonic degree. If the mean of the functions
    is zero, this is equivalent to the covariance of the two functions.

    ------------------------------------------------------------------------------------------------
    Copyright (C) <2023>  <Yanmichel A. Morfa>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import re
from datetime import date

import numeric_tools
import numpy as np
import pint
import xarray as xr

earth_radius = 6.3712e6  # Radius of Earth [m]

_global_attrs = {'grid': 'spectral',
                 'source': 'git@github.com:deterministic-nonperiodic/SEBA.git',
                 'institution': 'Max Planck Institute for Meteorology',
                 'history': date.today().strftime('Created on %c'),
                 'Conventions': 'CF-1.6'}

# Create a pint UnitRegistry object
UNITS_REG = pint.UnitRegistry()

# from Metpy
cmd = re.compile(r'(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])')


def _parse_units(unit_str):
    return UNITS_REG(cmd.sub('**', unit_str))


def kappa_from_deg(ls, linear=False):
    """
        Returns total horizontal wavenumber [radians / meter]
        from spherical harmonics degree (ls) on the surface
        of a sphere of radius Re using the Jeans formula.
        κ = sqrt[l(l + 1)] / Re ~ l / Re  for l>>1
    """
    num = ls if linear else np.sqrt(ls * (ls + 1.0))
    return num / earth_radius


def lambda_from_deg(ls, linear=False):
    """
    Returns wavelength λ [meters] from total horizontal wavenumber
    λ = 2π / κ
    """
    return 2.0 * np.pi / kappa_from_deg(ls, linear=linear)


def deg_from_lambda(lb):
    """
        Returns wavelength λ [meters] from spherical harmonics degree (ls)
    """
    deg = np.sqrt(0.25 + (2.0 * np.pi * earth_radius / lb) ** 2)
    return np.floor(deg - 0.5).astype(int)


def kappa_from_lambda(lb):
    return 2.0 * np.pi / lb


def cross_spectrum(clm1, clm2=None, lmax=None, convention='power', axis=0):
    """Returns the cross-spectrum of the spherical harmonic coefficients as a
    function of spherical harmonic degree.

    Signature
    ---------
    array = cross_spectrum(clm1, clm2, [degrees, lmax, convention, axis])

    Parameters
    ----------
    clm1 : ndarray, shape ((ntrunc+1)*(ntrunc+2)/2, ...)
        contains the first set of spherical harmonic coefficients.
    clm2 : ndarray, shape ((ntrunc+1)*(ntrunc+2)/2, ...), optional
        contains the second set of spherical harmonic coefficients.
    convention : str, optional, default = 'power'
        The type of spectrum to return: 'power' for power spectrum, 'energy'
        for energy spectrum, and 'l2norm' for the l2-norm spectrum.
    lmax : int, optional,
        triangular truncation
    axis : int, optional
        axis of the spectral coefficients

    Returns
    -------
    array : ndarray, shape (lmax + 1, ...)
        contains the cross spectrum as a function of spherical harmonic degree.
    """
    if convention not in ['energy', 'power']:
        raise ValueError("Parameter 'convention' must be one of"
                         " ['energy', 'power']. Given {}".format(convention))

    if clm2 is not None:
        msg = f"Arrays 'clm1' and 'clm2' must have the same shape. Expected shape: {clm1.shape}"
        assert clm2.shape == clm1.shape, msg

    # spectral coefficients moved to axis 0 for clean vectorized operations
    clm_shape = list(clm1.shape)
    nlm = clm_shape.pop(axis)

    # flatten sample dimensions
    clm1 = np.moveaxis(clm1, axis, 0).reshape((nlm, -1))

    # Get indexes of the triangular matrix with spectral coefficients
    truncation = numeric_tools.truncation(nlm)

    if lmax is not None:
        # If lmax is given make sure it is consistent with the number of clm coefficients
        truncation = min(lmax + 1, truncation)

    # Compute cross spectrum from spherical harmonic expansion coefficients as
    # a function of spherical harmonic degree (total wavenumber)
    if clm2 is None:
        spectrum = numeric_tools.cross_spectrum(clm1, clm1, truncation)
    else:
        clm2 = np.moveaxis(clm2, axis, 0).reshape((nlm, -1))
        spectrum = numeric_tools.cross_spectrum(clm1, clm2, truncation)

    if convention.lower() == 'energy':
        spectrum *= 4.0 * np.pi

    return np.moveaxis(spectrum.reshape(tuple([truncation] + clm_shape)), 0, axis)


def convert_to_complex(dataset, dim='nc2'):
    """Converts all variables in a xarray dataset to complex by using the first
    index along 'nc2' as the real part and the second as the imaginary part.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to convert to complex.
    dim : str,
        dimension along which to merge

    Returns
    -------
    xarray.Dataset
        The converted dataset.
    """

    # cast array to complex along an axis with dimension 2. The first index correspond to
    # the real part and the second index to the imaginary part in the resulting array.
    def cast_complex(arr, axis=0):
        res = 1j * np.take(arr, 1, axis=axis)
        res += np.take(arr, 0, axis=axis)
        return res

    # take the real and imaginary parts of spectral coefficients
    return dataset.reduce(cast_complex, dim=dim, keep_attrs=True)


def dataset_spectra(dataset, variables=None, truncation=None, convention='energy', dim_name='nsp'):
    """
    Computes power/energy spectrum of all variables in dataset

    :param dataset: xarray.Dataset,
        A dataset containing the complex spectral coefficients of the variables
    :param variables: list,
        list of variables to analyze
    :param truncation: int,
        spectral truncation
    :param convention: str,
        option to compute 'energy' or 'power' spectrum
    :param dim_name: str,
        dimension along which to compute spectra
    :return: xarray.Dataset,
        A new dataset containing the spectrum of each variable in 'dataset'
    """
    if variables is None:
        variables = list(dataset.data_vars)
    elif isinstance(variables, (list, tuple)):
        variables = [variable for variable in variables if variable in dataset.data_vars]
    else:
        raise ValueError("Unknown type for 'variables', must be an iterable of strings")

    if not variables:
        raise ValueError(f"Could not find variables {variables} in dataset")

    if truncation is None:
        truncation = numeric_tools.truncation(dataset.dims[dim_name])

    kappa_h = kappa_from_deg(np.arange(truncation, dtype=int))

    # save data units before computing spectrum
    var_units = {name: dataset[name].attrs.get("units") for name in variables}

    dataset = xr.apply_ufunc(
        cross_spectrum,
        dataset[variables],
        input_core_dims=[[dim_name, ]],  # dimension along which the spectrum is computed
        output_core_dims=[['kappa']],  # new wavenumber dimension
        exclude_dims={dim_name},  # dimensions allowed to change size.
        kwargs=dict(convention=convention, lmax=truncation - 1, axis=-1),
        dask="parallelized",
        output_dtypes=[np.complex64],  # map outputs to complex, then convert to real
        dask_gufunc_kwargs={'output_sizes': {'kappa': truncation}},
        keep_attrs=True
    )

    # assign new coordinate with horizontal wavenumber
    dataset = dataset.astype(np.float64).assign_coords({"kappa": kappa_h})

    # recover and square units for all variables after calculation
    for name, units in var_units.items():
        if units:
            converted_units = _parse_units(units) ** 2
            dataset[name].attrs['units'] = str(converted_units.units)

    # add attributes
    dataset.attrs.update(_global_attrs)
    dataset.attrs['truncation'] = f"TL{truncation - 1}"

    dataset.kappa.attrs = {'standard_name': 'wavenumber',
                           'long_name': 'horizontal wavenumber',
                           'axis': 'X', 'units': 'm**-1'}

    return dataset.transpose(..., 'kappa')


def process_files(file_path, output_path, variables=None, truncation=None):
    if variables is not None:
        if isinstance(variables, str):
            variables = variables.split(',')
        else:
            raise ValueError("Variables must be a string of comma-separated variable names.")

    # open dataset from files
    dataset = xr.open_mfdataset(file_path)

    # Cast dataset's variables to complex along dimension 'nc2' (consistent with CDO coefficients)
    dataset = convert_to_complex(dataset, dim='nc2')

    # Compute power spectrum for all variables in dataset up to a given truncation.
    # The default is nlat - 1, where nlat is the number of latitude points.
    dataset = dataset_spectra(dataset, variables=variables,
                              truncation=truncation,
                              convention='power')

    # export to netcdf file
    dataset.to_netcdf(output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="input files with spectral coefficients")
    parser.add_argument("-o", "--output", help="output file name")
    parser.add_argument("-t", "--truncation", type=int, help="triangular truncation")
    parser.add_argument("-select", "--variables", help="compute spectra of these variables")

    args = parser.parse_args()

    # Compute power spectra and export to netcdf dataset 'output_path'
    process_files(args.files, output_path=args.output,
                  variables=args.variables,
                  truncation=args.truncation)


if __name__ == '__main__':
    main()
