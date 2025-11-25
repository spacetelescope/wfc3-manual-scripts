# cal_ir_monitor_calspec
The WFC3/IR staring mode monitoring pipeline for standard stars.

## Description

The scripts in this repository are part of the pipeline to use staring mode observations of standard stars to monitor the photometric stability of the WFC3/IR detector.

This pipeline was used to reduce and analyze standard star staring data observed from installation to 2023. The methodology and results are discussed in [WFC3 ISR 2024-06](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2024/WFC3-ISR-2024-06.pdf): Time-Dependent Sensitivity of the WFC3/IR Detector (2009-2023) (Marinelli, et al.).

## Requirements

The pipeline requires the packages [`pyql`](https://github.com/spacetelescope/pyql), maintained by the HST/WFC3 Quicklook team, and [`wfc3-phot-tools`](https://github.com/spacetelescope/wfc3-phot-tools/), maintained by Mariarosa Marinelli. Please see the `cal_ir_monitor_calspec.yaml` file for all dependencies.

## Set Up

First, set up the environment and activate it.

```
conda env create -f cal_ir_monitor_calspec.yaml

conda activate cal_ir_monitor_calspec
```

Next, follow the repository instructions for installing the [`pyql`](https://github.com/spacetelescope/pyql) and [`wfc3-phot-tools`](https://github.com/spacetelescope/wfc3-phot-tools/) packages. 


## Usage

This pipeline is primarily designed to run from the command line, with a total of 21 possible configurable arguments: 3 pipeline settings, 6 pipeline execution flags, and 9 pipeline parameters.

    > python cal_ir_monitor_calspec.py 
          [-n NAME] [--trial] [--local]
          [--get_new_data] [--redownload] 
          [--helium] [--linearity] 
          [--run_ap_phot] [--plot_sources]
          [--nlinfile NLINFILE] 
          [--proposals PROPOSALS [PROPOSALS ...]]
          [--targets TARGETS [TARGETS ...]]
          [--filters FILTERS [FILTERS ...]]
          [--radius RADIUS] [--annulus ANNULUS] 
          [--dannulus DANNULUS] [--back_method {mean,median,mode}]
          [-w WRITE_DIR] 

The 18 arguments are explained in greater detail in `ir_phot_toolbox.py`, and can also be viewed by using the `--help` flag.

    > python cal_ir_monitor_calspec.py --help


## Code details

### `ir_download.py`
Functions to enable downloading of IR standard star staring mode calibration data.

### `ir_file_io.py`
Functions to manage file I/O, including checking and creating directories, filtering and moving files, and setting paths.

### `ir_fits.py`
Functions for handling FITS files, including easily accessing header information and extensions.

### `ir_helium_corr.py`
Enables helium correction in the F105W and F110W filters for the IR staring mode standard star pipeline.

### **`cal_ir_monitor_calspec.py`**

As the name suggests, this is the primary script for this pipeline. It can be run either from the command line or from a notebook environment.

The main function is `run_pipeline()`, which takes parameters of `args` and `dirs`.
- `args` are the pipeline arguments, which can be constructed one of two ways:
  - parsed from command line arguments/flags via the `parse_args()` function from `ir_toolbox.py`
  - set in a notebook via an `InteractiveArgs` class object from `ir_toolbox.py`
- `dirs` is a dictionary of four full directory paths ('data', 'bad', 'output', 'plots'), and is set by the `initialize_directories()` function in `ir_toolbox.py`.

Currently, the pipeline is not equipped to drizzle observations or perform photometry on drizzled files.

### `ir_plotting.py`
Functions to create plots for the IR standard star staring mode photometry monitor.

### `ir_syn.py`
Functions and a class to enable synthetic photometry for the IR staring mode standard star pipeline.

### `ir_td.py`
Functions to correct photometry for IR time-dependence, as defined by a calculated slope of relative sensitivity change over time.

### `ir_toolbox.py`
Assorted tools for the IR staring mode standard star pipeline.


---

Author: Mariarosa Marinelli
Contact: mmarinelli@stsci.edu
