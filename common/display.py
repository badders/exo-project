from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import aplpy
import matplotlib.pyplot as plt
from astropy.io import fits
import Tkinter

def show_fits(data, **kwargs):
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig = aplpy.FITSFigure(data, figure=fig)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray'
    fig.show_colorscale(aspect='auto', **kwargs)
    fig.ticks.hide()
    #fig.tight_layout()


def show_header(filename):
    master = Tkinter.Tk()
    scrollbar = Tkinter.Scrollbar(master, orient=Tkinter.VERTICAL)
    listbox = Tkinter.Listbox(master, yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=Tkinter.RIGHT, fill=Tkinter.Y)
    listbox.pack(side=Tkinter.LEFT, fill=Tkinter.BOTH, expand=1)

    h = fits.open(filename)[0].header
    for k in h:
        if k.strip() != '':
            listbox.insert(Tkinter.END, '{}: {}'.format(k, h[k]))

    Tkinter.mainloop()