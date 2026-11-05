"""Transport invariants for the x86 Athena bridge; no accuracy claims or hardware required."""
import ctypes as ct
import unittest
import numpy as np
from athena_acquisition import AthenaInlet, sample_counters
from lsl_acquisition import StreamAdapter


def packet(indices):
    indices = np.asarray(indices)
    x = np.zeros((len(indices), 30), dtype=np.float32)
    x[:, 0], x[:, 1] = indices // 256, indices % 256
    x[:, 6:] = np.arange(24)
    return x


class BridgeTests(unittest.TestCase):
    def test_epoch_boundary_preserves_sample_spacing(self):
        np.testing.assert_array_equal(sample_counters(packet([254, 255, 256, 257])), [254, 255, 256, 257])

    def test_bad_counters_are_rejected(self):
        for indices in ([1, 1], [3, 2]):
            with self.assertRaises(ValueError):
                sample_counters(packet(indices))
        x = packet([0, 1])
        x[0, 1] = .5
        with self.assertRaises(ValueError):
            sample_counters(x)
        x[0, 1] = np.nan
        with self.assertRaises(ValueError):
            sample_counters(x)

    def test_named_mapping_scales_only_eeg_fields(self):
        labels = ['Epoch', 'Offset', 'Hour', 'Min', 'Sec', 'MilliSec'] + [str(i) for i in range(24)]
        metadata = dict(sfreq=256, channel_count=30, labels=labels, units=['']*6 + ['uV']*24)
        adapter = StreamAdapter(metadata, dict(sfreq=256, channels=['19', '0', '8']))
        out, _ = adapter.transform(packet([0, 1]), [10., 10. + 1/256], 10. + 1/256)
        np.testing.assert_allclose(out[:, 0], np.array([19, 0, 8])*1e-6, rtol=1e-6)

    def test_gap_and_stale_detection_remain_active(self):
        metadata = dict(sfreq=256, channel_count=1, labels=['C3'], units=['V'])
        for times, now in (([1., 1.2], 1.2), ([1., 1.+1/256], 3.)):
            adapter = StreamAdapter(metadata, dict(sfreq=256, channels=['C3']))
            with self.assertRaises(ValueError):
                adapter.transform([[0.], [0.]], times, now)

    def test_sdk_clock_is_anchored_once_without_hiding_a_gap(self):
        inlet = AthenaInlet.__new__(AthenaInlet)
        inlet.pending = np.empty((0, 30), np.float32)
        inlet.pending_times = np.empty(0)
        inlet.anchor = inlet.last_counter = None
        clock = iter([10., 10.5])
        inlet.clock = lambda: next(clock)
        buffers = [(ct.c_float * 60)(*packet(ids).ravel()) for ids in ([0, 1], [130, 131])]
        iterator = iter(buffers)
        def read(size):
            ct.cast(size, ct.POINTER(ct.c_int))[0] = 2
            return next(iterator)
        inlet.raw_data = read
        inlet._read()
        initial_anchor = inlet.anchor
        inlet._read()
        self.assertEqual(inlet.anchor, initial_anchor)
        np.testing.assert_allclose(np.diff(inlet.pending_times), [1/256, 129/256, 1/256])

    def test_duplicate_block_is_not_retimed(self):
        inlet = AthenaInlet.__new__(AthenaInlet)
        inlet.pending = np.empty((0, 30), np.float32)
        inlet.pending_times = np.empty(0)
        inlet.anchor, inlet.last_counter, inlet.clock = None, 5, lambda: 10.
        buffer = (ct.c_float * 60)(*packet([4, 5]).ravel())
        def read(size):
            ct.cast(size, ct.POINTER(ct.c_int))[0] = 2
            return buffer
        inlet.raw_data = read
        with self.assertRaisesRegex(ValueError, 'counter reset'):
            inlet._read()


if __name__ == '__main__':
    unittest.main()
