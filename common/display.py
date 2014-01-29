from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import aplpy
import matplotlib.pyplot as plt
from astropy.io import fits
try:
    import Tkinter as tkinter
except:
    import tkinter
import numpy as np

def show_fits(data, **kwargs):
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig = aplpy.FITSFigure(data, figure=fig)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray'
    fig.show_colorscale(aspect='auto', **kwargs)
    fig.ticks.hide()
    #plt.ylim(tuple(reversed(plt.ylim())))
    return fig


def show_header(filename):
    master = tkinter.Tk()
    scrollbar = tkinter.Scrollbar(master, orient=tkinter.VERTICAL)
    listbox = tkinter.Listbox(master, yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
    listbox.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)

    h = fits.open(filename)[0].header
    for k in h:
        if k.strip() != '':
            listbox.insert(tkinter.END, '{}: {}'.format(k, h[k]))

    tkinter.mainloop()