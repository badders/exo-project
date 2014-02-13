# coding : utf-8
"""
Check file dates to see if a collection of source files is newer than an
output file
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import glob

def update_required(dest, pathname):
    try:
        dest_time = os.stat(dest).st_mtime
    except OSError:
        return True

    if isinstance(pathname, basestring):
        files = glob.glob(pathname)
    else:
        files = pathname

    if files is None:
        return False

    for fn in files:
        if os.stat(fn).st_mtime >= dest_time:
            return True

    return False