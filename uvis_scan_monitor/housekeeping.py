"""
tk - adjust

move check_subdirectory to wfc3_phot_tools ??? 
"""
from datetime import datetime

def _make_timestamp():
    """
    get rid of this, can be replaced by

        today = datetime.now()
        timestamp = today.strftime('%Y-%m-%d_%H-%M-S')

    Helper function to convert and format current datetime
    into a string. This string is then used for the name
    of the pipeline run directory in the scan monitor
    photometry central store location:
        /grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor

    Returns
    -------
    timestamp : str
        String representation of current time, in format
        `YYYY-MM-DD_hh-mm-ss`.
    """
    now = str(datetime.now()).split(' ')
    date = now[0]
    time = now[1].split('.')[0].replace(':', '-')
    timestamp = f'{date}_{time}'
    return timestamp

def check_subdirectory(parent_dir, sub_name):
    sub_dir = os.path.join(parent_dir, sub_name)
    if os.path.exists(sub_dir):
        print(f'Found existing directory at {sub_dir}')
    else:
        print(f'Making new directory at {sub_dir}...')
        os.mkdir(sub_dir)

    return sub_dir
