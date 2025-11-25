"""
Logging functions and classes for the IR staring mode
standard star photometry monitor pipeline.

Functions
---------
check_preexisting_logging()
    Checks if logging is already enabled.
note(verbose, log, log_type, message)
    Prints and/or logs some message.
make_timestamp()
    Creates string timestamp for current datetime.
setup_logging(local, log_dir, log_name, verbose, log)
    Configures pipeline logging for new run.

Classes
-------
CaptureOutput()

"""
from datetime import datetime
from io import StringIO
import logging
import os
import sys

from ir_config import CONFIG


def check_preexisting_logging():
    """Checks if logging is already enabled.

    Helper function to verify that no logging is already
    set up, so that when this is run in interactive mode,
    there is no confusion about where the logs are.

    Returns
    -------
    preexisting_logging : Boolean
        Whether or not there is any preexisting logging
        configured in the current session.
    """
    existing_handlers = logging.getLogger().handlers
    if len(existing_handlers) == 0:
        preexisting_logging = False

    else:
        # Then logging is already enabled for another file
        note(verbose=True, log=True, log_type='critical',
             message='Logging has already been enabled for file: '\
                     f'{str(existing_handlers[0]).split(" ")[1]}')
        preexisting_logging = True

    return preexisting_logging


def command_line_logging(args):
    """
    Parameters
    ----------
    args : Boolean
        Whether to log output for this pipeline run.
    """
    if args.log:
        log_dir = os.path.join(CONFIG['monitor_dir'], 'logs')
        setup_logging(log_dir=log_dir, log_name=args.name)
        logging.info('Logging initialized')


def note(verbose, log, message, log_type='info'):
    """Prints and/or logs some message.

    Parameters
    ----------
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    message : str
        Message to be displayed.
    log_type : str
        Logging message type; defaults to `info`. Should
        be `info`, `warning`, `error`, or `critical`;
        otherwise, displays an additional warning message
        and logs original message as `info` type.
    """
    if log:
        if log_type == 'info':
            logging.info(message)

        elif log_type == 'warning':
            logging.warning(message)

        elif log_type == 'error':
            logging.error(message)

        elif log_type == 'critical':
            logging.critical(message)

        else:
            log_type_message = '`note()` called with invalid '\
                               f'`log_type` = {log_type}\n'\
                               'Logging the following as `info` message:'
            logging.warning(log_type_message)
            logging.info(message)

    if verbose:
        print(message)





def setup_logging(local=False, log_dir=os.getcwd(),
                  log_name=make_timestamp(),
                  verbose=True, log=True):
    """Configures pipeline logging for new run.

    If pipeline is being run from the command line, this
    should be initialized at the beginning. Otherwise, this
    can be initialized when creating an `InteractiveArgs`
    object.

    Parameters
    ----------
    local : Boolean
        Whether to save the log locally (default) or to
        central storage log location.
    log_dir : str or path-like
        String representation or path to location where
        log should be saved. If no path is specified
        and `local` is set to `True`, then log will
        save to current working directory.
    log_name : str
        What to name the log; defaults to timestamp for
        moment of function execution.
    verbose : Boolean
        Whether to print the message; defaults to True.
    log : Boolean
        Whether to log the message; defaults to True.
    """
    if local:
        if not os.path.exists(log_dir):
            log_dir = os.getcwd()
    else:
        log_dir = os.path.join(CONFIG['monitor_dir'], 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            note(verbose=verbose, log=log, log_type='info',
                 message=f'Made logging directory at {log_dir}')

    log_file = os.path.join(log_dir, log_name)

    logging.basicConfig(filename=f'{log_file}.log', filemode='w',
                        format='%(asctime)s - %(levelname)-8s - %(message)s',
                        level=logging.INFO)

    note(verbose=verbose, log=log, log_type='info',
         message=f'Logging enabled. Writing to file: {log_file}.log')
