"""Pinned, bounded step algorithms. SI units; no fitting or device calibration.

B0 reproduces the frozen legacy core. B1 uses SKDH CWT components with a legal
low-pass filter and a documented fixed-scale fallback for short bouts. B2 calls
the official norm-based Shin detector, without tuning a timestamp offset.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt, resample_poly

METHODS = ('B0_legacy', 'B1_vertical_cwt', 'B2_shin')


def runs(mask):
    d = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def detect_core(accel_mps2, fs, method):
    x = np.asarray(accel_mps2, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError('finite Nx3 SI acceleration required')
    if not np.isfinite(fs) or not 20 <= fs <= 200:
        raise ValueError('engineering input range 20-200 Hz')
    if len(x) < max(8, int(np.ceil(.5 * fs))):
        return np.empty(0, int), 'TOO_SHORT'
    if np.max(np.ptp(x, axis=0)) < 1e-10:
        return np.empty(0, int), 'CONSTANT_SIGNAL'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', UserWarning)
        if method == 'B0_legacy':
            from skdh.gait_old.get_gait_events import get_gait_events
            ic, *_ = get_gait_events(x / 9.80665, fs, np.arange(len(x)) / fs,
                                     max(round(.2 / (1.25 / fs)), 1), 4, 20., True, True)
            return np.unique(ic.astype(int)), 'OK'
        if method == 'B1_vertical_cwt':
            from skdh.gait.substeps import PreprocessGaitBout, VerticalCwtGaitEvents
            g = x / 9.80665
            p = PreprocessGaitBout(filter_cutoff=min(20., .4 * fs))
            if len(x) >= int(np.ceil(5 * fs)):
                try:
                    kw = p.predict(time=np.arange(len(x)) / fs, accel=g, fs=fs)
                except ValueError as e:
                    if 'autocovariance' not in str(e):
                        raise
                    kw = None
            else:
                kw = None
            state = 'OK'
            if kw is None:
                # No arbitrary minimum of 8s. Use the official fixed CWT scale
                # when there is too little evidence for its adaptive frequency.
                va = int(np.argmax(np.abs(np.mean(g, axis=0))))
                filtered = detrend(sosfiltfilt(butter(4, min(20., .4 * fs), fs=fs,
                    btype='low', output='sos'), g, axis=0), axis=0)
                kw = dict(accel_filt=filtered, v_axis=va,
                          v_axis_sign=float(np.sign(np.mean(g[:, va])) or 1),
                          mean_step_freq=None)
                state = 'FIXED_SCALE_INSUFFICIENT_ADAPTIVE_CONTEXT'
            out = VerticalCwtGaitEvents(round_scale=False).predict(
                time=np.arange(len(x)) / fs, accel=g, fs=fs, **kw)
            return np.unique(np.asarray(out['initial_contacts'], int)), state
        if method == 'B2_shin':
            import pandas as pd
            from mobgap.initial_contacts import IcdShinImproved
            frame = pd.DataFrame(x, columns=['acc_x', 'acc_y', 'acc_z'])
            # mobgap's frame validator requires these column names even though
            # the norm detector reads acceleration only. NaN explicitly means
            # unobserved; these columns are never used by this algorithm.
            for key in ['gyr_x', 'gyr_y', 'gyr_z']:
                frame[key] = np.nan
            ic = IcdShinImproved(axis='norm').detect(frame, sampling_rate_hz=fs).ic_list_['ic']
            return np.unique(np.asarray(ic, int)), 'OK'
    raise ValueError(f'unknown method: {method}')


def detect_regions(x, fs, regions, method, context_s=2., block_s=30., valid=None):
    """Use real context, score only original regions; each event counted once.

    The fixed 30s COMPUTATION blocks have no meaning as walking bouts or pauses.
    Validity gaps break context. Returned event coordinates refer to the input.
    """
    x = np.asarray(x, float)
    valid = np.isfinite(x).all(axis=1) if valid is None else np.asarray(valid, bool) & np.isfinite(x).all(axis=1)
    requested = np.zeros(len(x), bool)
    for a, b in regions:
        if not 0 <= a <= b <= len(x):
            raise ValueError('region outside recording')
        requested[a:b] = True
    found, diagnostics = [], []
    context = int(round(context_s * fs)); block = max(1, int(round(block_s * fs)))
    for valid_a, valid_b in runs(valid):
        for a0, b0 in runs(requested[valid_a:valid_b]):
            a0 += valid_a; b0 += valid_a
            for a in range(a0, b0, block):
                b = min(a + block, b0)
                lo, hi = max(valid_a, a - context), min(valid_b, b + context)
                try:
                    ic, status = detect_core(x[lo:hi], fs, method)
                    ic = ic + lo
                    found.extend(ic[(ic >= a) & (ic < b)].tolist())
                    diagnostics.append(dict(start=int(a), end=int(b), status=status))
                except (ValueError, IndexError) as exc:
                    diagnostics.append(dict(start=int(a), end=int(b), status='FAILED', error=str(exc)))
    return np.unique(np.asarray(found, int)), diagnostics


def skdh_gait_regions(x, fs):
    """Independent fixed pretrained gait gate, evaluated as a complete workflow.

    Explicitly resample first to20Hz then pass20Hz, so the classifier's filter
    and its feature grid use the same sampling frequency. No fitted calibration.
    Three-second classifier windows do not bridge invalid input.
    """
    from fractions import Fraction
    from skdh.context import PredictGaitLumbarLgbm
    x = np.asarray(x, float)
    regions, meta = [], []
    for a, b in runs(np.isfinite(x).all(axis=1)):
        if b - a < 3 * fs:
            meta.append(dict(start=int(a), end=int(b), status='UNDER_THREE_SECONDS'))
            continue
        ratio = Fraction(20 / fs).limit_denominator(1000)
        y = resample_poly(x[a:b] / 9.80665, ratio.numerator, ratio.denominator, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            out = PredictGaitLumbarLgbm().predict(time=np.arange(len(y)) / 20., accel=y, fs=20.)
        for lo, hi in zip(out['Gait Bout Start Index'], out['Gait Bout Stop Index']):
            aa = a + int(np.rint(lo / 20 * fs)); bb = min(b, a + int(np.rint(hi / 20 * fs)))
            if aa < bb:
                regions.append((aa, bb))
        tail = (len(y) % 60) / 20
        meta.append(dict(start=int(a), end=int(b), status='OK', unclassified_tail_s=tail))
    return regions, meta


def ionescu_gait_regions(x, fs):
    """Official deterministic trunk gate, short-bout setting frozen before test.

    No merging of distinct gait segments across pauses. Adaptive thresholds are
    the stock inference algorithm, not a fitted/device-specific calibration.
    """
    import pandas as pd
    from mobgap.gait_sequences import GsdAdaptiveIonescu
    x = np.asarray(x, float)
    regions, meta = [], []
    for a, b in runs(np.isfinite(x).all(axis=1)):
        if b-a < 3*fs:
            meta.append(dict(start=int(a),end=int(b),status='UNDER_THREE_SECONDS'))
            continue
        frame=pd.DataFrame(x[a:b],columns=['acc_x','acc_y','acc_z'])
        for key in ['gyr_x','gyr_y','gyr_z']:frame[key]=np.nan
        detector=GsdAdaptiveIonescu(min_n_steps=3,max_gap_s=0.)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            detector.detect(frame,sampling_rate_hz=fs)
        for row in detector.gs_list_.itertuples():
            aa=max(int(a),int(a+row.start));bb=min(int(b),int(a+row.end))
            if aa<bb:regions.append((aa,bb))
        meta.append(dict(start=int(a),end=int(b),status='OK',
                         adaptive_threshold=float(detector.threshold_),
                         adaptive_threshold_success=bool(detector.adaptive_threshold_success_)))
    return regions,meta
