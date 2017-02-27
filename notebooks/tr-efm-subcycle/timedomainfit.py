# -*- coding: utf-8 -*-
"""
Time Domain Fit
===============

Equations to fit the frequency shift transient in the time domain
are developed here.
"""
from __future__ import division, print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sklearn
from scipy import linalg, signal, optimize
import lockin
import phasekick2 as pk2
import phasekick as pk
import tqdm
import sigutils
import munch

def avg_xt(tr, xr):
    tr_cent = (tr - tr.mean()) * 1e6 # Use µs units here
    A = np.c_[tr_cent ** 2, tr_cent, np.ones_like(tr_cent)]
    val, err, __, ___ = linalg.lstsq(A, xr)
    return val[-1], np.sqrt(err / (xr.size - 3)) / np.sqrt(xr.size)


def record_fit(f, xdata, ydata, p0=None, bounds=(-np.inf, np.inf), name=''):
    fit_data = munch.Munch()
    fit_data.p0 = p0
    fit_data.popt, fit_data.pcov = optimize.curve_fit(f, xdata, ydata, p0=p0, bounds=bounds)
    fit_data.x = f(xdata, *fit_data.popt)
    fit_data.resid = ydata - fit_data.x
    fit_data.name = name
    return fit_data

def _remove_harmonics(b, popt, x_har, N_harmonics=5):
    b.t_zeroed = b.t_ms - b.t_ms[0]
    b.A = harN(b.t_zeroed, popt[2], N_harmonics, np.exp(-b.t_zeroed * popt[3]))
    b.x_no_overtones = b.x - np.dot(b.A, x_har)
    b.resid = b.x_no_overtones - damped_sine(b.t_zeroed, *popt)

def remove_harmonics(b, bdamp):
    _remove_harmonics(b, bdamp.fit.popt, bdamp.x_har, bdamp.N_harmonics)

# def avg_trig_xt(tr, xr):
#     tr_cent = (tr - tr.mean()) * 1e6 # Use µs units here
#     A = np.c_[np.cos(0.062*2*np.pi*tr_cent),
#               np.sin(0.062*2*np.pi*tr_cent),
#               np.ones_like(tr_cent)]
#     val, err, __, ___ = linalg.lstsq(A, xr)
#     return (val[0] + val[-1]), np.sqrt(err / (xr.size - 3)) / np.sqrt(xr.size)

def damped_sine(t, X, Y, f, k, x0):
    """Sine wave with variable phase (X, Y), frequency f,
    damping rate k, and DC offset x0."""
    return np.exp(-k * t) *  (X * np.cos(2*np.pi*f*t) + Y * np.sin(2*np.pi*f*t)) + x0

def _damped_sine_exp_A(t, X, Y, f, k, x0, df, k_tau, k1):
    return np.where(t > 0,
        np.exp(-k1 * t),
        np.exp(-k * t))

def _damped_sine_exp_phi(t, X, Y, f, k, x0, df, k_tau, k1):
    return np.where(t > 0,
        2*np.pi*(f - df * np.expm1(-k_tau*t)) * t,
        2*np.pi*f*t
            )

def damped_sine_exp(t, X, Y, f, k, x0, df, k_tau, k1):
    phi = _damped_sine_exp_phi(t, X, Y, f, k, x0, df, k_tau, k1)
    A = _damped_sine_exp_A(t, X, Y, f, k, x0, df, k_tau, k1)
    return A * (X * np.cos(phi) + Y*np.sin(phi)) + x0

def harN(t, f, N, A=1):
    """Generate harmonics of a given frequency for least squares fitting."""
    cols = []
    for i in xrange(2, N+1):
        cols.extend([A*np.cos(t*2*np.pi*f*i), A*np.sin(t*2*np.pi*f*i)])
    return np.c_[cols].T

def harNphi(phi, N, A=1):
    """Generate harmonics of a given frequency for least squares fitting."""
    cols = []
    for i in xrange(2, N+1):
        cols.extend([A*np.cos(phi*i), A*np.sin(phi*i)])
    return np.c_[cols].T


def fit_damped_sine_eliminate_harmonics(t, x, p0, N_harmonics=5):
    b = munch.Munch()
    # Copy input parameters
    b.t = t
    b.t_zeroed = b.t - b.t[0]
    b.x = x
    b.N_harmonics = N_harmonics

    b.fit0 = record_fit(damped_sine, b.t_zeroed, x, p0=p0, name='initial fit')

    b.A = harN(b.t_zeroed, b.fit0.popt[2], N_harmonics, A=np.exp(-t * b.fit0.popt[3]))

    b.Afit = linalg.lstsq(b.A, b.fit0.resid)
    b.x_har = b.Afit[0]

    b.x_no_overtones = b.x - np.dot(b.A, b.x_har)
    b.fit = record_fit(damped_sine, b.t_zeroed, b.x_no_overtones, p0=p0, name='final fit')

    return b

def signal_average_gr(gr, ti, tf):
    """Utility function to signal average a group from an HDF5 file."""
    b = munch.Munch()
    xs = []
    ts = []
    for ds in tqdm.tqdm(gr.values()):
        t = pk.gr2t(ds)
        t1 = ds.attrs['Abrupt BNC565 CantClk.t1 [s]']
        t0 = -t1 - ds["half periods [s]"][0]
        m = (t > ti) & (t < tf)
        ts.append(t[m])
        x = ds['cantilever-nm'][:]
        xs.append(x[m])

    ts = np.array(ts)
    b.t = np.mean(ts, axis=0)
    x_array = np.array(xs)
    b.x = np.mean(x_array, axis=0)
    m2 = b.t < 0.0
    b.x = b.x - b.x[m2].mean()
    b.t_ms = b.t * 1e3
    b.t_us = b.t * 1e6
    b.t5 = b.t[500:]
    b.x5 = b.x[500:]
    b.t5ms = b.t5*1e3
    b.t5us = b.t5*1e6
    
    b.li = lockin.LockIn(b.t, b.x, 1e6)
    b.li.lock2()
    b.li.phase(tf=0)
    b.li.name='data'
    
    return b


def signal_average_parab_list(gr_list, ti, tf, invert=True, align_voltage=False):
    """Utility function to signal average a group from an HDF5 file."""
    b = munch.Munch()
    xs = []
    ts = []
    if invert:
        scale = -1
    else:
        scale = 1
    for ds in tqdm.tqdm(gr_list):
        # Move the zero in time to the initial voltage pulse
        x = ds['cantilever-nm'][:]
        dt = ds['dt [s]'].value

        if align_voltage:
            t1 = ds.attrs['Abrupt BNC565 CantClk.t1 [s]']
            t0 = -t1 - ds["half periods [s]"][0]
            t = np.arange(x.size)*dt + t0
        else:
            t = pk.gr2t(ds)

        m = (t > ti) & (t < tf)
        ts.append(t[m])
        x = ds['cantilever-nm'][:]
        xs.append(scale * x[m])

    ts = np.array(ts)
    b.t = np.mean(ts, axis=0)
    xs = np.array(xs)
    
    # Do proper signal averaging, fitting data to a parabola at each point
    x_err = np.zeros(xs.shape[1])
    x_bf = np.zeros(xs.shape[1])
    for i, (tr, xr) in tqdm.tqdm(enumerate(zip(ts.T, xs.T))):
        x, err = avg_xt(tr, xr)
        x_err[i] = err
        x_bf[i] = x

    b.x = x_bf
    b.x_err = x_err
    m2 = b.t < 0.0
    b.x0neg = b.x[m2].mean()
    b.x = b.x - b.x0neg
    b.t_ms = b.t * 1e3
    b.t_us = b.t * 1e6
    b.t5 = b.t[500:]
    b.x5 = b.x[500:]
    b.t5ms = b.t5*1e3
    b.t5us = b.t5*1e6
    
    b.li = lockin.LockIn(b.t, b.x, 1e6)
    b.li.lock2()
    b.li.phase(tf=0)
    b.li.name='data'
    
    return b


def signal_average_gr_parab(gr, ti, tf, invert=True, align_voltage=False):
    """Utility function to signal average a group from an HDF5 file."""
    return signal_average_parab_list(gr.values(), ti, tf, invert=True, align_voltage=False)

def pk_phase(f_i, df, f_f, tau, t0, tp):
    return lambda t: 2 * np.pi * np.where(t <= t0, f_i * (t-t0),
                      np.where(t < tp, f_i*(t-t0)+ pk.phase_step(t-t0, tau, df),
                              f_i*(tp-t0)+ pk.phase_step(tp-t0, tau, df) + f_f*(t-tp)))

def pk_freq(f_i, df, f_f, tau, t0, tp):
    return lambda t: np.where(t <= t0, f_i,
                      np.where(t < tp, f_i - df *np.expm1(-(t-t0)/tau), f_f)
                            )

def osc(phi, amp, X0, Y0):
    return (X0 * np.cos(phi) + Y0 * np.sin(phi)) * amp

def osc_phase(t, phase, A, X0, Y0):
    return osc(phase(t), A(t), X0, Y0)

def osc_freq(t, freq, A, X0, Y0):
    dt = np.r_[0, np.diff(t)]
    phi = np.cumsum(freq(t) * dt)
    return osc(phi, A(t), X0, Y0)

def getA(ki, km, kf, ti, tf):
    def A(t):
        t_ = t - ti
        Af = np.exp(-km*(tf-ti))
        return np.where(t <= ti, np.exp(-ki * t_), np.where(t < tf,
                        np.exp(-km * t_), Af * np.exp(-(t-tf)*kf)))
    return A

def get_xDCt(phaset, ft, A, tau, xDC0, dx_light, t0, tp):
        delta = (ft(tp) / ft(t0))**2
        r = 1 - delta
        
        omega_tau = (2*np.pi*ft(t0)*tau)
        omega_tau2 = omega_tau**2
        omega_bar = (phaset(tp) - phaset(t0)) / (tp - t0)
        xeq = lambda t: np.where(t <= t0, xDC0, np.where(t < tp, xDC0-dx_light*np.expm1(-(t-t0)/tau),0))
        
        xresp = lambda t: np.where(t <= t0, xDC0, np.where(t < tp, r*(
                xDC0 + dx_light -
                dx_light * omega_tau2 / (1+omega_tau2) * np.exp(-(t-t0)/tau)
                ) + 
                delta*xDC0*np.cos(omega_bar*(t-t0)) -
                dx_light * r /(1+omega_tau2) * (
                    np.cos(omega_bar * (t-t0)) + omega_tau*np.sin(omega_bar * (t-t0))
                ),np.nan))

        xDC = lambda t: (xresp(t) - xeq(t)) * A(t)/A(t0) + xeq(t)
        return xDC

def make_simple_opt(t0, tp, Q, f0, f_HP, f_LP, fs):
    """A variety of simple fitting functions."""
    b, a = signal.butter(2, np.array([f_HP, f_LP]) / (fs/2), analog=False, btype='bandpass')
    
    k_ringdown = 2*np.pi*f0 / Q
    def simp_opt(t, f_i, df, f_f, X0, Y0, tau):
        """Initial frequency f_i, light-induced frequency shift df,
        final frequency f_f, initial phase X0, Y0, time constant tau.

        Use the fixed ringdown time k_ringdown."""
        phi = pk_phase(f_i, df, f_f, tau, t0, tp)(t)
        amp = np.exp(-k_ringdown*t)
        return osc(phi, amp, X0, Y0)

    simp_opt.A = lambda t, f_i, df, f_f, X0, Y0, tau: np.exp(-k_ringdown*t)
    simp_opt.phi = lambda t, f_i, df, f_f, X0, Y0, tau: pk_phase(f_i, df, f_f, tau, t0, tp)(t)

    def simp_opt2(t, f_i, df, f_f, X0, Y0, tau1, tau2, ratio):
        """
        Parameters:
            f_i: initial frequency
             df: light-induced frequency shift
            f_f: final frequency
             X0: initial phase (cosine term)
             Y0: initial phase (sine term)
           tau1: fast time constant
           tau2: slow time constant
          ratio: fraction of df associated with fast time constant tau1

        Fixed:
            k_ringdown: single dissipation inverse time constant

     """
        df1 = ratio * df
        df2 = (1 - ratio) * df
        amp = np.exp(-k_ringdown*t)
        phi = (pk_phase(f_i, df1, f_f, tau1, t0, tp)(t)
               + pk_phase(0.0, df2, 0.0, tau2, t0, tp)(t)
               )
        return osc(phi, amp, X0, Y0)

    simp_opt2.A = lambda t, f_i, df, f_f, X0, Y0, tau1, tau2, ratio: np.exp(-k_ringdown*t)
    simp_opt2.phi = lambda t, f_i, df, f_f, X0, Y0, tau1, tau2, ratio: (
        pk_phase(f_i, ratio * df, f_f, tau1, t0, tp)(t)
               + pk_phase(0.0, (1 - ratio) * df, 0.0, tau2, t0, tp)(t)
        )


    def simp_optDC(t, f_i, df, f_f, X0, Y0, tau1, tau2, ratio, Q0, Qlight, dx_light):
        """
        Parameters:
                f_i: initial frequency
                 df: light-induced frequency shift
                f_f: final frequency
                 X0: initial phase (cosine term)
                 Y0: initial phase (sine term)
               tau1: fast time constant
               tau2: slow time constant
              ratio: fraction of df associated with fast time constant tau1
                 Q0: Initial quality factor (before light)
             Qlight: Quality factor under illumination
           dx_light: light induced change in position


        Fixed:
            k_ringdown: single dissipation inverse time constant
            """


        df1 = ratio * df
        df2 = (1 - ratio) * df
        k0 = 2*np.pi*f0 /Q0
        klight = 2*np.pi*f0 / Qlight
        # A, phase, freq vs time.
        A = getA(k0, klight, k0, t0, tp)

        phase = lambda t: (pk_phase(f_i, df1, f_f, tau1, t0, tp)(t)
               + pk_phase(0.0, df2, 0.0, tau2, t0, tp)(t))

        freq = lambda t: (pk_freq(f_i, df1, f_f, tau1, t0, tp)(t)
               + pk_freq(0.0, df2, 0.0, tau2, t0, tp)(t))

        xDC = get_xDCt(phase, freq, A, tau1, 0.0, dx_light, t0, tp)

        return osc_phase(t, phase, A, X0, Y0) + xDC(t)

    def simp_optDCphi(t, f_i, df, f_f, X0, Y0, tau1, tau2, ratio, Q0, Qlight, dx_light):
        """
        Parameters:
                f_i: initial frequency
                 df: light-induced frequency shift
                f_f: final frequency
                 X0: initial phase (cosine term)
                 Y0: initial phase (sine term)
               tau1: fast time constant
               tau2: slow time constant
              ratio: fraction of df associated with fast time constant tau1
                 Q0: Initial quality factor (before light)
             Qlight: Quality factor under illumination
           dx_light: light induced change in position


        Fixed:
            k_ringdown: single dissipation inverse time constant
            """


        df1 = ratio * df
        df2 = (1 - ratio) * df
        k0 = 2*np.pi*f0 /Q0
        klight = 2*np.pi*f0 / Qlight
        # A, phase, freq vs time.
        A = getA(k0, klight, k0, t0, tp)

        phase = lambda t: (pk_phase(f_i, df1, f_f, tau1, t0, tp)(t)
               + pk_phase(0.0, df2, 0.0, tau2, t0, tp)(t))

        freq = lambda t: (pk_freq(f_i, df1, f_f, tau1, t0, tp)(t)
               + pk_freq(0.0, df2, 0.0, tau2, t0, tp)(t))

        xDC = get_xDCt(phase, freq, A, tau1, 0.0, dx_light, t0, tp)

        return osc_phase(t, phase, A, X0, Y0) + xDC(t)

    def _simp_optDCDet_A(t, f_i, df, f_f, X0, Y0, tau1, tau2, ratio, Q0, Qlight, dx_light):
        df1 = ratio * df
        df2 = (1 - ratio) * df
        k0 = 2*np.pi*f0 /Q0
        klight = 2*np.pi*f0 / Qlight
        return getA(k0, klight, k0, t0, tp)(t)

    def _simp_optDCDet_phi(t, f_i, df, f_f, X0, Y0, tau1, tau2, ratio, Q0, Qlight, dx_light):
        df1 = ratio * df
        df2 = (1 - ratio) * df
        k0 = 2*np.pi*f0 /Q0
        klight = 2*np.pi*f0 / Qlight
        return (pk_phase(f_i, df1, f_f, tau1, t0, tp)(t)
               + pk_phase(0.0, df2, 0.0, tau2, t0, tp)(t))

    def simp_optDCDet(t, f_i, df, f_f, X0, Y0, tau1, tau2, ratio, Q0, Qlight, dx_light):
        df1 = ratio * df
        df2 = (1 - ratio) * df
        k0 = 2*np.pi*f0 /Q0
        klight = 2*np.pi*f0 / Qlight
        # A, phase, freq vs time.
        A = getA(k0, klight, k0, t0, tp)

        phase = lambda t: (pk_phase(f_i, df1, f_f, tau1, t0, tp)(t)
               + pk_phase(0.0, df2, 0.0, tau2, t0, tp)(t))

        freq = lambda t: (pk_freq(f_i, df1, f_f, tau1, t0, tp)(t)
               + pk_freq(0.0, df2, 0.0, tau2, t0, tp)(t))

        xDC = get_xDCt(phase, freq, A, tau1, 0.0, dx_light, t0, tp)

        xout = osc_phase(t, phase, A, X0, Y0) + xDC(t)
        return signal.lfilter(b, a, xout)[500:]

    simp_optDCDet.A = _simp_optDCDet_A
    simp_optDCDet.phi = _simp_optDCDet_phi

    return simp_opt, simp_opt2, simp_optDCDet


def fit(f, xdata, ydata, p0=None, bounds=(-np.inf, np.inf), tfit=None, name=''):
    fit_data = munch.Munch()
    fit_data.popt, fit_data.pcov = optimize.curve_fit(f, xdata, ydata, p0=p0, bounds=bounds)
    fit_data.x = f(xdata, *fit_data.popt)
    fit_data.resid = ydata - fit_data.x
    if tfit is None:
        tfit = xdata
    
    fs = 1.0/np.mean(np.diff(tfit))
    li = lockin.LockIn(tfit, fit_data.x, fs)
    li.lock2()
    li.phase(tf=0)
    fit_data.li = li
    fit_data.name = name
    fit_data.li.name = name
    return fit_data


def make_optfunc(t0, tp, f_LP=200e3, f_HP=1e3, fs=1e6, Q=30000):
    """This huge function currently implements all of the various curve fitting options developed."""

    b, a = signal.butter(2, np.array([f_HP, f_LP]) / (fs/2), analog=False, btype='bandpass')

    def phit(t, f_i, df_light, f_f, tau):
        t = t - t0
        tp_ = tp - t0
        return 2*np.pi*np.where(t <= 0, f_i*t,
                      np.where(t < tp_, f_i*t+ pk.phase_step(t, tau, df_light),
                              f_i*tp_+ pk.phase_step(tp_, tau, df_light) + f_f*(t-tp_)))
    
    def get_ft(f_i, df, f_f, tau):
        def ft(t):
            t = t - t0
            tp_ = tp - t0
            return np.where(t <= 0, f_i,
                      np.where(t < tp_, f_i - df_light *np.expm1(-t/tau), f_f)
                            )

    
    def xDCt(t, phaset, ft, tau, xDC0, dx_light, A):
        t_ = t - t0
        tp_ = tp - t0
        
        delta = (ft(tp) / ft(t0))**2
        r = 1 - delta
        
        omega_tau = (2*np.pi*ft(t0)*tau)
        omega_tau2 = omega_tau**2
        omega_bar = (phaset(tp) - phaset(t0)) / (tp - t0)
        xeq = lambda t: np.where(t <= t0, xDC0, np.where(t < tp, xDC0-dx_light*np.expm1(-(t-t0)/tau),0))
        
        xresp = lambda t: np.where(t <= t0, xDC0, np.where(t < tp, ))

        return (r*(
                xDC0 + dx_light -
                dx_light * omega_tau2 / (1+omega_tau2) * np.exp(-t_/tau)
                ) + 
                delta*xDC0*np.cos(omega_bar*t) -
                dx_light * r /(1+omega_tau2) * (
                    np.cos(omega_bar * t) + omega_tau*np.sin(omega_bar * t)
                ))

    def vDCt(t, f_i, df_light, f_f, tau, xDC0, dx_light):
        t = t - t0
        tp_ = tp - t0
        delta = (f_f / f_i)**2
        r = 1 - delta
        omega_tau = (2*np.pi*f_i*tau)
        omega_tau2 = omega_tau**2
        omega_bar = 2*np.pi * (
                        phit(tp, f_i, df_light, f_f, tau) -
                         phit(t0, f_i, df_light, f_f, tau)
                        ) / (tp - t0)
        return (r*dx_light*omega_tau/(1+omega_tau2)*np.exp(-t/tau) -
                2*np.pi*delta*f_i* xDC0 * np.sin(omega_bar*t) -
                dx_light * r * omega_bar /(1+omega_tau2) * (
                    np.sin(omega_bar * t) - omega_tau*np.cos(omega_bar * t)
                ))
    
    
    def xDC_all(t, f_i, df_light, f_f, X0, Y0, tau, xDC0, dx_light):
        t_ = t - t0
        tp_ = tp - t0
        k_ringdown = 2*np.pi*f_i / Q
        
        phi = phit(t, f_i, df_light, f_f, tau)
        return np.where(t_ < 0, xDC0,
                      np.where(t_ < tp_,
                        xDCt(t, f_i, df_light, f_f, tau, xDC0, dx_light),
                        (xDCt(tp, f_i, df_light, f_f, tau, xDC0, dx_light) * np.cos(2*np.pi*f_f*(t-tp)) + 
                        vDCt(tp, f_i, df_light, f_f, tau, xDC0, dx_light) * np.sin(2*np.pi*f_f*(t-tp)) / (2*np.pi*f_f)
                         ) * np.exp(-t*k_ringdown)
                              ))
        
    def optfuncDC(t, f_i, df_light, f_f, X0, Y0, tau, Q0, Qp, xDC0, dx_light,):
        t_ = t - t0
        tp_ = tp - t0
        k_ringdown = 2*np.pi*f_i / Q
        
        phase = pk_phase(f_i, df_light, f_f, tau, t0, tp)
        freq = pk_freq(f_i, df_light, f_f, tau, t0, tp)
        
        k_ringdown = 2*np.pi*f_i / Q0
        k_light = 2*np.pi*f_i / Qp
        Ap = np.exp(-k_light * (tp-t0))
        A = getA(k_ringdown, k_light, k_ringdown, t0, tp)
        
        xDC = np.where(t_ < 0, xDC0,
                      np.where(t_ < tp_,
                        (xDCt(t, phase, freq, tau, xDC0, dx_light, A) - xDC0)*np.exp(-k_light*t)+xDC0,
                        (xDCt(t, phase, freq, tau, xDC0, dx_light, A) * np.cos(2*np.pi*f_f*(t-tp)) + 
                         vDCt(tp, f_i, df_light, f_f, tau, xDC0, dx_light) * np.sin(2*np.pi*f_f*(t-tp)) / (2*np.pi*f_f)
                         ) * Ap * np.exp(-k_ringdown * (t-tp))
                              ))

        xout = osc_phase(t, phase, A, X0, Y0) + xDC
        return signal.lfilter(b, a, xout)[500:]
    
    def optfuncDC2(t, f_i, df_light, f_f, X0, Y0, tau, Q0, Qp, xDC0, dx_light, tau2, ratio):
        t_ = t - t0
        tp_ = tp - t0
        k_ringdown = 2*np.pi*f_i / Q
        df1 = ratio * df_light
        df2 = (1-ratio) * df_light
        phi = phit(t, f_i, df1, f_f, tau) + phit(t, 0.0, df2, 0.0, tau2)
        
        
        k_ringdown = 2*np.pi*f_i / Q0
        k_light = 2*np.pi*f_i / Qp
        Ap = np.exp(-k_light * (tp-t0))
        
        xDC = np.where(t_ < 0, xDC0,
                      np.where(t_ < tp_,
                        (xDCt(t, f_i, df_light, f_f, tau, xDC0, dx_light) - xDC0)*np.exp(-k_light*t)+xDC0,
                        (xDCt(tp, f_i, df_light, f_f, tau, xDC0, dx_light) * np.cos(2*np.pi*f_f*(t-tp)) + 
                         vDCt(tp, f_i, df_light, f_f, tau, xDC0, dx_light) * np.sin(2*np.pi*f_f*(t-tp)) / (2*np.pi*f_f)
                         ) * Ap * np.exp(-k_ringdown * (t-tp))
                              ))
        xout = ((X0 * np.cos(phi) + Y0 * np.sin(phi))*np.where(t < 0, np.exp(-t*k_ringdown),
                                                                np.where(t < tp, np.exp(-k_light*t),
                                                                        Ap * np.exp(-k_ringdown * (t-tp))))
                + xDC)
        return signal.lfilter(b, a, xout)[500:]

    
    def optfunc(t, f_i, df_light, f_f, X0, Y0, tau):
        t = t - t0
        k_ringdown = 2*np.pi*f_i / Q
        phi = phit(t, f_i, df_light, f_f, tau)
        return (X0 * np.cos(phi) + Y0 * np.sin(phi))*np.exp(-t*k_ringdown)
    
    def optfuncDet(t, f_i, df_light, f_f, X0, Y0, tau, Q0, Qp):
        t = t - t0
        k_ringdown = 2*np.pi*f_i / Q0
        k_light = 2*np.pi*f_i / Qp
        phi = phit(t, f_i, df_light, f_f, tau)
        Ap = np.exp(-k_light * (tp-t0))
        xout = (X0 * np.cos(phi) + Y0 * np.sin(phi))*np.where(t < 0, np.exp(-t*k_ringdown),
                                                                np.where(t < tp, np.exp(-k_light*t),
                                                                        Ap * np.exp(-k_ringdown * (t-tp))))
        return signal.lfilter(b, a, xout)[500:]

    def optfuncQ(t, f_i, df_light, f_f, X0, Y0, tau, Q0, Qp):
        t = t - t0
        k_ringdown = 2*np.pi*f_i / Q0
        k_light = 2*np.pi*f_i / Qp
        phi = phit(t, f_i, df_light, f_f, tau)
        Ap = np.exp(-k_light * (tp-t0))
        return (X0 * np.cos(phi) + Y0 * np.sin(phi)) * np.where(t < 0, np.exp(-t*k_ringdown),
                                                                np.where(t < tp, np.exp(-k_light*t),
                                                                        Ap * np.exp(-k_ringdown * (t-tp))))

    return phit, optfunc, optfuncQ, optfuncDC, optfuncDet, xDCt, xDC_all