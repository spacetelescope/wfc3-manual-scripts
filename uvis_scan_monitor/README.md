This notebook shows how to perform aperture photometry on calibrated (FLT) WFC3/UVIS spatial scan observations in order to monitor the instrument sensitivity over time.

By the end of this tutorial, you will:

- Locate and download new spatial scan observations from MAST.
- Verify data quality of spatial scans, discarding any compromised exposures.
- Remove cosmic rays from exposures and perform aperture photometry on the spatial scans.
- Produce intermediate photometry catalogs for each target and filter.
- Clip and normalize data to produce plots.

Dependencies:

Two WFC3 packages must be installed in your conda environment before beginning the notebook.

1. `wfc3_phot_tools` (tools for UVIS photometric calibration originally developed by Claire Shanahan; maintained by Mariarosa Marinelli)
    - From the command line, run:
        git clone https://github.com/spacetelescope/wfc3-phot-tools`
    - Add the cloned WFC3_phot_tools directory to your \\$PATH and/or \\$PYTHONPATH by adding export command to shell configuration files (ex. `.bash_profile` or `.bashrc`).
    - Enter cloned directory and install tools with
        python setup.py install

2. `pyql` (quicklook tools developed and maintained by WFC3 Quicklook team at STScI)
    - From the command line, run:
        git clone https://github.com/spacetelescope/pyql
    - Add the cloned pyql directory to your \\$PATH and/or \\$PYTHONPATH by adding export command to shell configuration files (ex. `.bash_profile` or `.bashrc`)
    - The credentials to access QL database are not available through GitHub, so request a copy of `config.yaml` from someone with QL access. Then, move it into the `pyql/pyql/pyql_settings/` directory, replacing the original cloned version.
    - Enter cloned directory and install tools with:
        python setup.py install
