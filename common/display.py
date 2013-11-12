from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import aplpy
import matplotlib.pyplot as plt


def show_fits(data, **kwargs):
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig = aplpy.FITSFigure(data, figure=fig)
    fig.show_grayscale(aspect='auto', **kwargs)
    fig.ticks.hide()
    #fig.tight_layout()
