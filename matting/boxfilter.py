import numpy as np
from ctypes import CDLL, c_int, c_double, POINTER
from .load_libmatting import load_libmatting

library = load_libmatting()

c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)

_boxfilter = library.boxfilter
_boxfilter.argtypes = [
    c_double_p,
    c_int,
    c_int,
    c_double_p,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
]


def get_boxfilter_params(src, r, mode):
    ms, ns = src.shape

    modes = {
        "valid": 0,
        "same": 1,
        "full": 2,
    }

    assert(mode in modes)

    mode = modes[mode]

    if mode == 0:
        md = ms - 2 * r
        nd = ns - 2 * r
        di = r
        dj = r
    elif mode == 1:
        md = ms
        nd = ns
        di = 0
        dj = 0
    elif mode == 2:
        md = ms + 2 * r
        nd = ns + 2 * r
        di = -r
        dj = -r

    return ms, ns, md, nd, di, dj, mode


def apply_to_channels(single_channel_func):
    def multi_channel_func(image, *args, **kwargs):
        if len(image.shape) == 2:
            return single_channel_func(image, *args, **kwargs)
        else:
            shape = image.shape
            image = image.reshape(shape[0], shape[1], -1)

            result = np.stack([
                single_channel_func(image[:, :, c].copy(), *args, **kwargs)
                for c in range(image.shape[2])], axis=2)

            return result.reshape(list(result.shape[:2]) + list(shape[2:]))

    return multi_channel_func


@apply_to_channels
def boxfilter(src, r, mode, dst=None):
    ms, ns, md, nd, di, dj, mode = get_boxfilter_params(src, r, mode)

    if dst is None:
        dst = np.empty((md, nd))

    assert(src.flags["C_CONTIGUOUS"])
    assert(src.flags["ALIGNED"])
    assert(src.dtype == np.float64)
    assert(dst.flags["C_CONTIGUOUS"])
    assert(dst.flags["ALIGNED"])
    assert(dst.dtype == np.float64)

    assert(dst.shape[0] == md)
    assert(dst.shape[1] == nd)

    _boxfilter(
        np.ctypeslib.as_ctypes(dst.ravel()),
        dst.shape[0],
        dst.shape[1],
        np.ctypeslib.as_ctypes(src.ravel()),
        src.shape[0],
        src.shape[1],
        di,
        dj,
        r,
        mode)

    return dst


def test(m, n, r, mode, n_runs=1):
    src = np.random.rand(m, n)
    kernel = np.ones((2 * r + 1, 2 * r + 1))
    kernel /= kernel.sum()

    import scipy.signal
    dst_ground_truth = scipy.signal.correlate2d(src, kernel, mode=mode)

    for _ in range(n_runs):
        t = time.perf_counter()

        dst = boxfilter(src, r, mode)

        dt = time.perf_counter() - t

        max_error = np.max(np.abs(dst - dst_ground_truth))

        print("%f gbyte/sec, %f seconds, max_error: %.20f %e %d-by-%d, r=%d" % (
            dst.nbytes * 1e-9 / dt, dt, max_error, max_error, m, n, r))

        assert(max_error < 1e-10)


def main():
    for mode in ["valid", "same", "full"]:

        for r in range(1, 5):
            min_size = 2 * r + 1
            for m in range(min_size, min_size + 10):
                for n in range(min_size, min_size + 10):
                    test(m, n, r, mode)

        test(512, 512, 4, mode, n_runs=20)


if __name__ == "__main__":
    main()
