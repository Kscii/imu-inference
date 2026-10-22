"""Frozen E7 feature definitions extracted from the audited public-trained chain.
No training, annotations, recording identity, research paths or caches are consumed.
Source identities are recorded in assets/e7/model.json.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d, minimum_filter1d, maximum_filter1d
from scipy.signal import butter, sosfiltfilt
from scipy.integrate import cumulative_trapezoid
from ..activity import engine
spans = engine.spans
RADII = [25, 50, 100]
GYRO_COLUMNS = np.r_[np.arange(174), [174+6*b+j for b in range(14) for j in [3,4,5]], np.arange(258,262)].astype(int)


def acc_features(x, valid, centers):
    """No labels used. Every operation stays inside a physical valid run."""
    assert np.all(valid[centers])
    z = engine.representation(x, valid)
    out = np.empty((len(centers), 174), np.float32)
    for lo, hi in spans(valid):
        at = np.flatnonzero((centers >= lo) & (centers < hi))
        if not len(at):
            continue
        ix = centers[at] - lo
        v = z[lo:hi, engine.ACC]
        local = [v[ix]]
        for n in [25, 101]:
            mu = uniform_filter1d(v, n, axis=0, mode="nearest")
            sd = np.sqrt(np.maximum(0, uniform_filter1d(v*v, n, axis=0, mode="nearest") - mu*mu))
            local.extend([mu[ix], sd[ix],
                          minimum_filter1d(v, n, axis=0, mode="nearest")[ix],
                          maximum_filter1d(v, n, axis=0, mode="nearest")[ix]])
        # Compute spectra only at requested centers, with the same physical edge clamp.
        freq = np.fft.rfftfreq(128, 1/25)
        band = (freq >= .5) & (freq <= 4.)
        for channel in [3, 4]:
            spectral = np.empty((len(ix), 5), np.float64)
            for offset in range(0, len(ix), 1024):
                chosen = ix[offset:offset+1024]
                ids = np.clip(chosen[:, None] + np.arange(-50, 51)[None, :], 0, len(v)-1)
                win = v[ids, channel].astype(np.float64)
                win -= win.mean(1, keepdims=True)
                en = np.sum(win*win, 1)
                ac = np.fft.irfft(np.abs(np.fft.rfft(win, n=256))**2, n=256)[:, 7:63]
                ac = np.clip(ac/np.maximum(en[:, None], 1e-12), -1, 1)
                k = ac.argmax(1)
                power = abs(np.fft.rfft(win*np.hanning(101), n=128))**2
                bp = power[:, band]
                peak = bp.argmax(1)
                f = np.column_stack([ac[np.arange(len(chosen)), k], 25/(k+7),
                                      bp.sum(1)/np.maximum(power.sum(1), 1e-12),
                                      bp[np.arange(len(chosen)), peak]/np.maximum(bp.sum(1), 1e-12),
                                      freq[band][peak]])
                f[en < 1e-6] = 0
                spectral[offset:offset+len(chosen)] = f
            local.append(spectral)
        assert sum(q.shape[1] for q in local) == 82
        raw = x[lo:hi, :3].astype(np.float64)/9.80665
        if len(raw) > 24:
            grav = sosfiltfilt(butter(2, .3, fs=25, output="sos"), raw, axis=0)
        else:
            grav = uniform_filter1d(raw, min(13, len(raw)), axis=0, mode="nearest")
        unit = grav/np.maximum(np.linalg.norm(grav, axis=1, keepdims=True), 1e-8)
        tr = np.column_stack([v, unit])
        smooth = uniform_filter1d(tr, 15, axis=0, mode="nearest")
        for offset in [-100, -50, -25, 0, 25, 50, 100]:
            local.append(smooth[np.clip(ix+offset, 0, len(raw)-1)])
        # Explicit before/after impulses and available context fraction.
        integ = np.r_[0., np.cumsum(v[:, 4])/25]
        for h in [25, 50, 100]:
            l = np.maximum(ix-h, 0)
            r = np.minimum(ix+h+1, len(raw))
            local.extend([(integ[ix+1]-integ[l])[:, None],
                          (integ[r]-integ[ix+1])[:, None],
                          ((ix-l)/h)[:, None], ((r-ix-1)/h)[:, None]])
        local.append((smooth[np.minimum(ix+100, len(raw)-1), :3]-
                      smooth[np.maximum(ix-100, 0), :3]))
        q = np.column_stack(local).astype(np.float32)
        assert q.shape == (len(ix), 174), q.shape
        out[at] = q
    assert np.isfinite(out).all()
    return out

def six_axis_features(x, valid, centers, observed=True):
    acc = acc_features(x, valid, centers)
    result = np.full((len(centers), 88), np.nan, np.float32)
    result[:, -4:] = 0
    if not observed:
        return np.column_stack([acc, result])
    z = engine.representation(x, valid)
    for lo, hi in spans(valid):
        at = np.flatnonzero((centers >= lo) & (centers < hi))
        if not len(at):
            continue
        ix = centers[at]-lo
        w = np.column_stack([x[lo:hi, 3:6], z[lo:hi, 6:9]])
        assert np.isfinite(w).all()
        f = [w[ix]]
        for n in [25, 101]:
            mu = uniform_filter1d(w, n, axis=0, mode="nearest")
            sd = np.sqrt(np.maximum(0, uniform_filter1d(w*w, n, axis=0, mode="nearest")-mu*mu))
            f.extend([mu[ix], sd[ix], maximum_filter1d(abs(w), n, axis=0, mode="nearest")[ix]])
        smooth = uniform_filter1d(w, 15, axis=0, mode="nearest")
        for offset in [-100, -50, -25, 0, 25, 50, 100]:
            f.append(smooth[np.clip(ix+offset, 0, hi-lo-1)])
        f.append(np.ones((len(ix), 1), np.float32))
        for radius in [12, 50, 100]:
            f.append(((np.minimum(ix+radius+1, hi-lo)-np.maximum(ix-radius, 0))/(2*radius+1))[:, None])
        result[at] = np.column_stack(f)
    assert np.isfinite(result).all()
    return np.column_stack([acc, result]).astype(np.float32)

def invariant_features(x, valid, centers):
    """Positive-scale and rigid-rotation invariant, label-blind, gap-isolated.

Acceleration is divided by local low-pass gravity magnitude. All remaining
features use norms/dot products, preserving signed motion along gravity.
No source or target normalization parameters are estimated and saved.
"""
    assert np.all(valid[centers])
    out = None
    for lo, hi in spans(valid):
        ids = np.flatnonzero((centers >= lo) & (centers < hi))
        if not len(ids):
            continue
        ix = centers[ids]-lo
        raw = np.asarray(x[lo:hi, :3], np.float64)
        g = (sosfiltfilt(butter(2, .3, fs=25, output="sos"), raw, axis=0)
             if len(raw) > 24 else uniform_filter1d(raw, 13, axis=0, mode="nearest"))
        gn = np.linalg.norm(g, axis=1)
        assert np.isfinite(raw).all() and (gn > 1e-12).all(), "Undefined local gravity ratio"
        u = g/gn[:, None]
        dyn = (raw-g)/gn[:, None]
        vertical = np.sum(dyn*u, axis=1)
        horizontal = np.linalg.norm(dyn-vertical[:, None]*u, axis=1)
        du = np.diff(u, axis=0, prepend=u[:1])*25
        da = np.diff(raw, axis=0, prepend=raw[:1])*25/gn[:, None]
        v = np.column_stack([vertical, horizontal, np.linalg.norm(dyn, axis=1),
                             np.linalg.norm(raw, axis=1)/gn-1,
                             np.linalg.norm(du, axis=1),
                             np.diff(vertical, prepend=vertical[:1])*25,
                             np.linalg.norm(da, axis=1),
                             np.diff(gn, prepend=gn[:1])*25/gn])
        cols = [v[ix]]
        for n in [13, 25, 51, 101]:
            mu = uniform_filter1d(v, n, axis=0, mode="nearest")
            var = uniform_filter1d(v*v, n, axis=0, mode="nearest")-mu*mu
            cols.extend([mu[ix], np.sqrt(np.maximum(0, var[ix])),
                         minimum_filter1d(v, n, axis=0, mode="nearest")[ix],
                         maximum_filter1d(v, n, axis=0, mode="nearest")[ix]])
        sm = uniform_filter1d(v, 13, axis=0, mode="nearest")
        for offset in [-100, -50, -25, 0, 25, 50, 100]:
            cols.append(sm[np.clip(ix+offset, 0, len(raw)-1)])
        integ = np.r_[0., np.cumsum(vertical)/25]
        for h in [25, 50, 100]:
            left, right = np.maximum(ix-h, 0), np.minimum(ix+h+1, len(raw))
            cols.extend([(integ[ix+1]-integ[left])[:, None],
                         (integ[right]-integ[ix+1])[:, None],
                         ((ix-left)/h)[:, None], ((right-ix-1)/h)[:, None]])
            # Cosine avoids angle instability near perfectly aligned gravity.
            cols.append(np.sum(u[left]*u[right-1], axis=1)[:, None])
        f = np.column_stack(cols)
        assert f.shape[1] == 207
        if out is None:
            out = np.empty((len(centers), f.shape[1]), np.float32)
        out[ids] = f.astype(np.float32)
    assert out is not None and np.isfinite(out).all()
    return out

def integration(x, valid, pos, gravity_unit=None):
    """78 descriptors divided by local gravity norm, with physical-gap resets.

    This is a label-blind feature ratio, not estimated sensor calibration.
    Scale is evaluated at each center and applied before numerical sign tests.
    VQF is used only with verified SI accelerometer/gyroscope inputs.
    """
    output = np.empty((len(pos), 78), np.float32)
    assert len(pos) and np.all(valid[pos])
    for lo, hi in spans(valid):
        rows = np.flatnonzero((pos >= lo) & (pos < hi))
        if not len(rows):
            continue
        raw = np.asarray(x[lo:hi, :3], float)
        n = len(raw)
        g = (sosfiltfilt(butter(2, .3, fs=25, output="sos"), raw, axis=0)
             if n > 24 else uniform_filter1d(raw, 13, axis=0, mode="nearest"))
        magnitude = np.linalg.norm(g, axis=1)
        assert np.isfinite(raw).all() and (magnitude > 1e-9).all()
        unit = g/magnitude[:, None] if gravity_unit is None else gravity_unit[lo:hi]
        # A constant offset vanishes under endpoint drift removal. Omitting it
        # avoids an unproven SI assertion for acceleration-only public exports.
        acc = np.column_stack([(raw*unit).sum(1), np.linalg.norm(raw, axis=1)])
        if n > 24:
            acc = sosfiltfilt(butter(4, 1.3, fs=25, output="sos"), acc, axis=0)
        vi = cumulative_trapezoid(acc, dx=.04, axis=0, initial=0)
        di = cumulative_trapezoid(vi, dx=.04, axis=0, initial=0)
        for offset in range(0, len(rows), 512):
            selected = rows[offset:offset+512]
            center = pos[selected]-lo
            columns, coverage = [], []
            for radius in RADII:
                request = center[:, None]+np.arange(-radius, radius+1)
                observed = (request >= 0) & (request < n)
                ix = np.clip(request, 0, n-1)
                left, right = ix[:, 0], ix[:, -1]
                elapsed = (ix-left[:, None])/25
                duration = np.maximum((right-left)/25, .04)
                drift = (vi[right]-vi[left])/duration[:, None]
                velocity = (vi[ix]-vi[left, None]-elapsed[:, :, None]*drift[:, None])
                displacement = (di[ix]-di[left, None]-elapsed[:, :, None]*vi[left, None]
                                -.5*elapsed[:, :, None]**2*drift[:, None])
                velocity /= magnitude[center, None, None]
                displacement /= magnitude[center, None, None]
                count = observed.sum(1)
                for channel in range(2):
                    v, d = velocity[:, :, channel], displacement[:, :, channel]
                    mean = (v*observed).sum(1)/count
                    std = np.sqrt(np.maximum(0, (v*v*observed).sum(1)/count-mean*mean))
                    peak = np.argmax(np.where(observed, abs(v), -np.inf), axis=1)
                    peak = np.where(np.where(observed, abs(v), 0).max(1) > 1e-6, peak, radius)
                    q = np.arange(len(selected))
                    columns.extend([v[:, radius], np.where(observed, v, np.inf).min(1),
                        np.where(observed, v, -np.inf).max(1), (abs(v)*observed).sum(1)/count,
                        std, v[q, peak], d[:, -1], d[:, radius],
                        np.where(observed, d, -np.inf).max(1)-np.where(observed, d, np.inf).min(1),
                        ((v > 1e-6) & observed).sum(1)/count, (ix[q, peak]-center)/25, mean])
                coverage.extend([(center-left)/radius, (right-center)/radius])
            output[selected] = np.column_stack(columns+coverage)
    assert np.isfinite(output).all()
    return output

def gravity_unit(raw):
    raw = np.asarray(raw[:, :3], np.float64)
    g = (sosfiltfilt(butter(2, .3, fs=25, output="sos"), raw, axis=0)
         if len(raw) > 24 else uniform_filter1d(raw, 13, axis=0, mode="nearest"))
    norm = np.linalg.norm(g, axis=1)
    assert np.isfinite(raw).all() and (norm > 1e-12).all()
    return g/norm[:, None]

def summarize(u, ix):
    columns = [u[ix]]
    for width in [25, 101]:
        mu = uniform_filter1d(u, width, axis=0, mode="nearest")
        sd = np.sqrt(np.maximum(0, uniform_filter1d(u*u, width, axis=0, mode="nearest")-mu*mu))
        columns.extend([mu[ix], sd[ix]])
    smooth = uniform_filter1d(u, 13, axis=0, mode="nearest")
    for offset in [-100, -50, -25, 0, 25, 50, 100]:
        columns.append(smooth[np.clip(ix+offset, 0, len(u)-1)])
    return np.column_stack(columns)

def shape_features(x, valid, pos):
    """72 columns: sorted abs gravity36 then signed sensor-axis gravity36.

    Both are scale-invariant. First36 also ignores axis signs/permutations,
    but intentionally preserves tilt within the housing. Neither is invariant
    to arbitrary rotations. No estimated mounting angle, labels or calibration.
    """
    assert len(pos) and np.all(valid[pos])
    out = np.empty((len(pos), 72), np.float32)
    for lo, hi in spans(valid):
        at = np.flatnonzero((pos >= lo) & (pos < hi))
        if not len(at):
            continue
        u = gravity_unit(x[lo:hi])
        ix = pos[at]-lo
        out[at] = np.column_stack([summarize(np.sort(abs(u), axis=1), ix), summarize(u, ix)])
    assert np.isfinite(out).all()
    return out
