from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import aplpy

def show_fits(data):
    fig = aplpy.FITSFigure(data)
    fig.show_grayscale(aspect='auto')
    #fig.tight_layout()
