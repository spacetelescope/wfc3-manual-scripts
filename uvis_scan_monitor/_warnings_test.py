import warnings

#: Developer specific warning are usually ignored when run from itnerpreter
#: Force it to be displayed always
warnings.simplefilter('always', DeprecationWarning)


def level_1_warning(not_reqd_arg=None):
    if not_reqd_arg is not None:
        warnings.warn("A level 1 warning", DeprecationWarning, stacklevel=1)
    return True

def level_2_warning(not_reqd_arg=None):
    if not_reqd_arg is not None:
        warnings.warn("A level 2 warning", DeprecationWarning, stacklevel=2)
    return True

def level_3_warning(not_reqd_arg=None):
    if not_reqd_arg is not None:
        warnings.warn("A level 3 warning", DeprecationWarning, stacklevel=3)
    return True


if __name__ == '__main__':
    level_1_warning("Not None")
    level_2_warning("Not None")
    level_3_warning("Not None")
