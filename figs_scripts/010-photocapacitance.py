# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
mpl.style.use("classic")
import pystan
import pickle
import h5py
import phasekick
import pmefm
import lockin
from scipy.optimize import curve_fit
from scipy import optimize
from tqdm import tqdm
from scipy import signal

def align_labels(axes_list,axis='y',align=None):
    if align is None:
        align = 'l' if axis == 'y' else 'b'
    yx,xy = [],[]
    for ax in axes_list:
        yx.append(ax.yaxis.label.get_position()[0])
        xy.append(ax.xaxis.label.get_position()[1])

    if axis == 'x':
        if align in ('t','top'):
            lim = max(xy)
        elif align in ('b','bottom'):
            lim = min(xy)
    else:
        if align in ('l','left'):
            lim = min(yx)
        elif align in ('r','right'):
            lim = max(yx)

    if align in ('t','b','top','bottom'):
        for ax in axes_list:
            t = ax.xaxis.label.get_transform()
            x,y = ax.xaxis.label.get_position()
            ax.xaxis.set_label_coords(x,lim,t)
    else:
        for ax in axes_list:
            t = ax.yaxis.label.get_transform()
            x,y = ax.yaxis.label.get_position()
            ax.yaxis.set_label_coords(lim,y,t)



filename = '../data/tr-efm/151217-200319-p1sun-df.h5'  
fh = h5py.File(filename, 'r')

fp = 1000
fc = 4000
tf = -0.052
tmin = -0.005
tmax = 0.150


li0 = phasekick.gr2lock(fh['data']['0000'], fp=fp, fc=fc)
li51 = phasekick.gr2lock(fh['data']['0000'], fp=fp/2, fc=fc/2)


filename = '../data/ancillary-efm/151218-021818-20sun-watch-decay-live.h5'  
fh2 = h5py.File(filename, 'r')

fp = 1000
fc = 4000
tf = -0.052
tmin = -0.01
tmax = 0.04

d2 = phasekick.AverageTrEFM.from_group(fh2['data'], fp, fc, tf, tmin, tmax)


size = 9
rcParams = {'figure.figsize': (2.0, 2.5), 'font.size': size,
#             'lines.markersize': ,
            'lines.linewidth': 1,
            'xtick.labelsize': size, 'ytick.labelsize': size,}



i = np.argmax(d2.tm_ms > 0)
phi = np.cumsum((d2.df50 + 155.6) *1e-6)
phi_off = (phi - phi[i])*1e3


with mpl.rc_context(rcParams):
    fig = plt.figure(figsize=(2.25, 2.))
    gs = gridspec.GridSpec(7,6)

    ax1 = fig.add_subplot(gs[0, 0:4])
    ax2 = fig.add_subplot(gs[1:4, 0:4])
    ax3 = fig.add_subplot(gs[-3:, 0:4])
    ax12 = fig.add_subplot(gs[0, -2:])
    ax22 = fig.add_subplot(gs[1:4, -2:])
    ax32 = fig.add_subplot(gs[-3:, -2:])
    
    gs.update(wspace=0.1, hspace=0.005) # set the spacing between axes. 
    ax2.plot(li0('t')*1e3, li0('df'), zorder=1)
    # plt.plot(li28.t, li28.df, zorder=0)
    ax1.plot(li51('t')*1e3, np.where(li51('t') > 0 , 1, 0), 'g')
    ax1.set_ylim(-0.4, 1.4)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels([0, r'$\quad\quad\:\:\:$'])
    ax2.plot(li51.t*1e3, li51.df, zorder=2, color='m', linewidth=1.5)
    ax2.set_ylim(-167, -143)
#    ax2.set_yticklabels(['', u'–165', u'–160', u'–155', u'–150', r'$\quad\quad\:\:\:$'], fontsize=9)
    # plt.plot((s.f['workup/time/x_rippleless'][:]+li0.t[0])[::200], np.gradient(s.f['workup/time/p'])[::200]/1e-6 - 61992,)
    ax3.set_xlabel("Time [ms]", fontsize=size)
    ax2.set_ylabel(r"$\delta f$ [Hz]", fontsize=size)
    ax1.set_xticklabels([''])
    ax1.set_ylabel(r'$I_{h\nu}$', fontsize=size)
    ax3.set_ylim(-200, 30)
    ax3.plot(li0.t*1e3, (li0.dphi+148*li0.t*2*np.pi+44.15)/(2*np.pi)*1000,)
    ax3.plot(li51.t*1e3, (li51.dphi+148*li51.t*2*np.pi+44.15)/ (2*np.pi)*1000, 'm--')

    ax1.set_xticks(np.linspace(-5, 20, 6))
    ax2.set_xticks(np.linspace(-5, 20, 6))
    ax2.set_xlim(-3, 20)
    ax3.set_xlim(-3, 20)
    ax1.set_xlim(-3, 20)
    ax1.set_xticklabels([''])
    ax2.set_xticklabels([''])
    ax3.set_yticks(np.arange(-200, 25, 50))
    ax3.set_yticklabels([u'–200', u'–150', u'–100', u'–50', u'0'])
    ax3.set_ylabel(r'$\delta \phi$ [mcyc.]', fontsize=size)
    ax3.set_xticklabels(['', 0, 5, 10, 15, ''])

    ax12.plot(d2.tm_ms, np.where(d2.tm_ms < 2.5, 1, 0), 'g-')
    ax12.set_ylim(-0.4, 1.4)
    ax12.set_xlim(0, 40)
    ax12.set_yticks([0, 1])
    ax12.set_yticklabels([''])
    ax12.set_xticklabels([''])
    ax22.plot(d2.tm_ms, d2.df50)
    ax22.set_xlim(0, 40)
    ax22.set_ylim(-167-10, -143-10)

    ax2.set_yticklabels(["", "", -160, "", -150, ""])
    ax3.set_yticks(np.arange(0, -201, -50))
    ax3.set_yticklabels([0, "", -100, "", -200])

    ax22.set_yticklabels([''])
    ax32.plot(d2.tm_ms[d2.tm_ms > 0], phi_off[d2.tm_ms > 0])
    ax32.set_xlim(0, 40)
    ax32.set_yticklabels([''])
    ax12.set_xticks(np.arange(0, 41, 20))
    ax22.set_xticks(np.arange(0, 41, 20))
    ax32.set_xticks(np.arange(0, 41, 20))
    ax22.set_xticklabels([''])
    ax32.set_xlabel("Time [ms]", fontsize=size)
    ax22.axhline(-156.5, color='k', linestyle='--')
    ax1.yaxis.set_label_coords(-0.37, 0.5)
    fig.savefig('../figs/01cd-photocapacitance.pdf',
                bbox_inches='tight', transparent=True)

