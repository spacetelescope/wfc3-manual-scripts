"""File handling including directory pathways, copying, moving, 
setting permissions, etc.

Authors
-------
    Aidan Pidgeon, 2024

Use
---

"""

import glob
import os
import shutil

import pandas as pd
from astropy.io import fits

from pyql.database.ql_database_interface import session
from pyql.database.ql_database_interface import Master, IR_flt_0, SingleFiles

def make_path_dict(parent_dir: str) -> dict:
    """Builds a dictionary containing absolute paths to be used
    throughout the pipeline.

    This function serves as a single place for hard-coded paths that
    are needed for file operations throughout the pipeline.  Some of
    paths are actually stings to be used in ``glob.glob()`` statements,
    (i.e. they contain wildcards).

    Parameters
    ----------
    parent_dir : str
        The absolute path of the main directory as determined by
        the ``-d --outdir`` argument of the ``cal_uvis_make_darks``
        module.

    Returns
    -------
    paths : dict
        A dictonary whose keys are path identifiers and whose
        values are strings containing absolute paths.
    """

    paths = {}

    # Hard-coded paths
    paths['raw_dir'] = '/grp/hst/wfc3i/ir_darks/data/raw_files/'
    paths['ima_dir'] = '/grp/hst/wfc3i/ir_darks/data/ima_files/'
    paths['spt_dir'] = '/grp/hst/wfc3i/ir_darks/data/spt_files/'
    paths['persist_dir'] = '/grp/hst/wfc3i/ir_darks/data/persist_files/'

    # Variable paths

    # Wildcard paths

    return paths

def query_darks(samp_seq: str, aper: str) -> pd.DataFrame:
    """Queries the WFC3 Quicklook database for all darks of a given 
    observing mode.
    
    Parameters
    ----------
    samp_seq : str
        The sample sequence to query for. Acceptable entries are:
        RAPID
        SPARS[5, 10, 25, 50, 100, 200]
        STEP[25, 50, 100, 200, 400] 
    aper : str
        The aperture to query for. The actual keyword used for 
        the query is SUBTYPE, which encompasses each subarray for a 
        given size, (e.g. SQ512SUB grabs both IRSUB512 and IRSUB512-FIX).
        Acceptable entries are:
        FULLIMAG
        SQ[64, 128, 256, 512]SUB 

    Returns
    -------
    query_results : Pandas DataFrame
        The results of the query contained in a Pandas DataFrame.
        Includes columns for rootname, directory, PID, as well as
        sample sequence and subtype for validation. 
    """ 
    # List of IR dark monitor program IDs through Cycle 30
    irdark_pids = [11929, 12097, 12349, 12695, 13077, 13562, 14008, 14374, 
                   14537, 14986, 15578, 15723, 16403, 16575, 17011]
    
    # pyql query
    pyql_results = session.query(Master.rootname, Master.dir, Master.link,
                                 IR_flt_0.proposid, IR_flt_0.samp_seq, IR_flt_0.subtype).\
                    join(IR_flt_0).\
                    filter(Master.detector == 'ir').\
                    filter(IR_flt_0.imagetyp == 'DARK').\
                    filter(IR_flt_0.samp_seq == samp_seq).\
                    filter(IR_flt_0.subtype == aper).\
                    all()
    
    # Convert query results to a DataFrame and filter for PID
    query_results = pd.DataFrame(pyql_results, columns=['rootname', 'dir', 'proposid,\
                                                         samp_seq, subtype'])
    query_results = query_results[query_results['proposid'].isin(irdark_pids)]

    return query_results


    