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

import logging
import re
from datetime import date

import numeric_tools
import numpy as np
import pint
import xarray as xr

earth_radius = 6.3712e6  # Radius of Earth [m]

_global_attrs = {
    'grid': 'spectral (spherical harmonic degrees)',
    'source': 'git@github.com:deterministic-nonperiodic/SEBA.git',
    'institution': 'Max Planck Institute for Meteorology',
    'history': date.today().strftime('Created on %c'),
    'Conventions': 'CF-1.6',
}

# Create a pint UnitRegistry object
UNITS_REG = pint.UnitRegistry()

# from Metpy
cmd = re.compile(r'(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])')

# module-level logger
logger = logging.getLogger(__name__)

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
    """Convert only data variables that *contain* ``dim`` from (real, imag) → complex.

    - Leaves coordinates and variables without ``dim`` unchanged (e.g., ``height_bnds``).
    - Requires ``dim`` length to be exactly 2.
    """

    if dim not in dataset.dims:
        raise ValueError(f"Dimension '{dim}' not found in dataset dims: {list(dataset.dims)}")
    if int(dataset.sizes[dim]) != 2:
        raise ValueError(
            f"Dimension '{dim}' must have length 2 (real, imag). Got size={dataset.sizes[dim]}"
        )

    def cast_complex(arr, axis=()):
        # xarray passes axis as a tuple for Dataset.reduce
        if isinstance(axis, tuple):
            if len(axis) == 0:
                return arr
            ax = axis[0]
        else:
            ax = axis
        real = np.take(arr, 0, axis=ax)
        imag = np.take(arr, 1, axis=ax)
        return real + 1j * imag

    # Only convert data variables that actually have the complex-pair dimension
    vars_with_dim = [name for name, da in dataset.data_vars.items() if dim in da.dims]
    if not vars_with_dim:
        logger.info("convert_to_complex: no data variables contain dim '%s' — nothing to convert.", dim)
        return dataset

    logger.info("convert_to_complex: converting variables along dim '%s': %s", dim, vars_with_dim)

    converted = dataset[vars_with_dim].reduce(cast_complex, dim=dim, keep_attrs=True)

    # Reassign converted variables back into the original dataset (others unchanged)
    return dataset.assign(**{name: converted[name] for name in vars_with_dim})


def dataset_spectra(dataset, variables=None, truncation=None, convention='energy', dim_name='nsp'):
    """
    Computes power/energy spectrum of selected variables in dataset.

    Only variables that contain the spectral dimension ``dim_name`` (e.g., ``'nsp'``)
    are eligible. Others (like bounds arrays) are ignored.
    """
    # Determine eligible variables
    eligible = [name for name, da in dataset.data_vars.items() if dim_name in da.dims]

    if variables is None:
        variables = eligible
    elif isinstance(variables, (list, tuple)):
        # Keep only variables explicitly requested *and* eligible
        variables = [v for v in variables if v in eligible]
    else:
        raise ValueError("Unknown type for 'variables', must be an iterable of strings")

    if not variables:
        raise ValueError(
            f"No variables with dimension '{dim_name}' found to compute spectra. "
            f"Eligible variables were: {eligible}"
        )

    if truncation is None:
        truncation = numeric_tools.truncation(dataset.sizes[dim_name])

    logger.info("dataset_spectra: computing '%s' spectrum for variables: %s (truncation=%s)",
                convention, variables, truncation)
   
    kappa_h = kappa_from_deg(np.arange(truncation, dtype=int))

    # save data units before computing spectrum
    var_units = {name: dataset[name].attrs.get("units") for name in variables}

    dataset_out = xr.apply_ufunc(
        cross_spectrum,
        dataset[variables],
        input_core_dims=[[dim_name]],  # dimension along which the spectrum is computed
        output_core_dims=[["kappa"]],  # new wavenumber dimension
        exclude_dims={dim_name},  # dimensions allowed to change size
        kwargs=dict(convention=convention, lmax=truncation - 1, axis=-1),
        dask="parallelized",
        output_dtypes=[np.complex64],  # map outputs to complex, then convert to real
        dask_gufunc_kwargs={"output_sizes": {"kappa": truncation}},
        keep_attrs=True,
        on_missing_core_dim="drop",  # ignore vars lacking the core dim (safety net)
    )

    # assign new coordinate with horizontal wavenumber
    dataset_out = dataset_out.astype(np.float64).assign_coords({"kappa": kappa_h})

    # recover and square units for all variables after calculation
    for name, units in var_units.items():
        # Update units → squared
        if units:
            converted_units = _parse_units(units) ** 2
            dataset_out[name].attrs["units"] = str(converted_units.units)
   
        # Add suffix to long_name and standard_name if present
        for key in ("long_name", "standard_name"):
            val = dataset[name].attrs.get(key)
            if isinstance(val, str) and val:
                name_meta = f"{val}_spectrum" if "standard" in key else f"{val} spectrum"
                dataset_out[name].attrs[key] = name_meta
   
    # add attributes
    dataset_out.attrs.update(_global_attrs)
    dataset_out.attrs["truncation"] = f"TL{truncation - 1}"

    dataset_out.kappa.attrs = {
        "standard_name": "wavenumber",
        "long_name": "horizontal wavenumber",
        "axis": "X",
        "units": "m**-1",
    }

    return dataset_out.transpose(..., "kappa")


def drop_cdi_grid_type(ds, vars_to_clean=None):
    """Remove the 'CDI_grid_type' attribute from selected data variables (or all).
    Returns a new Dataset with attributes removed.
    """
    if vars_to_clean is None:
        vars_to_clean = list(ds.data_vars)
    updates = {}
    for name in vars_to_clean:
        if "CDI_grid_type" in ds[name].attrs:
            v = ds[name].copy()
            v.attrs = {k: v.attrs[k] for k in v.attrs if k != "CDI_grid_type"}
            updates[name] = v
    return ds.assign(**updates)


def process_files(file_path, output_path, variables=None, truncation=None):
    if variables is not None:
        if isinstance(variables, str):
            variables = variables.split(',')
        else:
            raise ValueError("Variables must be a string of comma-separated variable names.")

    logger.info("Opening dataset(s): %s", file_path)

    # open dataset from files
    dataset = xr.open_mfdataset(file_path)

    # Cast dataset's variables to complex along dimension 'nc2' (consistent with CDO coefficients)
    dataset = convert_to_complex(dataset, dim='nc2')

    # Explicitly remove CDI_grid_type so downstream tools (e.g., CDO) don't infer spectral coeffs
    dataset = drop_cdi_grid_type(dataset)

    # Compute power spectrum for all variables in dataset up to a given truncation.
    # The default is nlat - 1, where nlat is the number of latitude points.
    dataset = dataset_spectra(dataset, variables=variables,
                              truncation=truncation,
                              convention='power')

    logger.info("Writing spectra to: %s", output_path)

    # export to netcdf file
    dataset.to_netcdf(output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="input files with spectral coefficients")
    parser.add_argument("-o", "--output", help="output file name")
    parser.add_argument("-t", "--truncation", type=int, help="triangular truncation")
    parser.add_argument("-select", "--variables", help="compute spectra of these variables")
    parser.add_argument("--log-level", default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        help="Logging verbosity (default: INFO)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # Compute power spectra and export to netcdf dataset 'output_path'
    process_files(args.files, output_path=args.output,
                  variables=args.variables,
                  truncation=args.truncation)


if __name__ == '__main__':
    main()
