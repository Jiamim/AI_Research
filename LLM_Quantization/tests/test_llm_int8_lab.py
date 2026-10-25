"""Independent numerical checks for the LLM.int8 teaching implementation."""

import unittest

import numpy as np

from experiments.llm_int8_lab import int8_matmul, quantize_absmax, split_matmul


class LlmInt8ReferenceTests(unittest.TestCase):
    def test_vector_scaling_and_integer_accumulation(self):
        x = np.array([[1., 1.], [2., 2.]])
        w = np.array([[3., 3.], [4., 4.], [-5., -5.]])
        result, state = int8_matmul(x, w)
        np.testing.assert_allclose(result, x @ w.T)
        self.assertEqual(state['accumulator'].dtype, np.int32)
        self.assertEqual(int(state['accumulator'][0, 0]), 2 * 127**2)
        self.assertEqual(state['sx'].shape, (2, 1))
        self.assertEqual(state['sw'].shape, (3, 1))

    def test_selects_column_union_and_removes_no_contribution(self):
        x = np.array([[.02, 30., .04], [.08, .1, -.03]])
        w = np.array([[.5, .2, -.4], [.8, -.6, .3]])
        result, info = split_matmul(x, w, threshold=6.)
        np.testing.assert_array_equal(info['outliers'], [False, True, False])
        # Even the small .1 from token 2 is routed with the selected feature column.
        np.testing.assert_allclose(info['y_outlier'], x[:, [1]] @ w[:, [1]].T)
        self.assertLess(np.max(np.abs(result - x @ w.T)), .001)

    def test_no_outliers_equals_vectorwise_and_all_outliers_equals_float(self):
        x = np.array([[2., -3.], [4., 5.]])
        w = np.array([[.12, .34], [-.56, .78]])
        none, info = split_matmul(x, w, threshold=10)
        np.testing.assert_array_equal(none, int8_matmul(x, w)[0])
        self.assertFalse(info['outliers'].any())
        all_high, info = split_matmul(x, w, threshold=1)
        np.testing.assert_array_equal(all_high, x @ w.T)
        self.assertIsNone(info['quantization'])

    def test_zero_row_and_half_step_error_bound(self):
        rng = np.random.default_rng(2)
        value = rng.normal(size=(10, 40))
        value[0] = 0
        codes, scale = quantize_absmax(value, axis=1)
        self.assertTrue(np.isfinite(scale).all())
        self.assertTrue((codes[0] == 0).all())
        self.assertTrue(np.all(np.abs(codes * scale - value) <= scale/2 + 1e-14))

    def test_invalid_values_and_thresholds(self):
        x = np.ones((2, 4))
        w = np.ones((3, 4))
        for threshold in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                split_matmul(x, w, threshold)
        for invalid in (np.ones(4), np.empty((0, 4)), np.array([[np.nan]*4]), x.astype(complex)):
            with self.assertRaises(ValueError):
                int8_matmul(invalid, w)
        with self.assertRaises(ValueError):
            int8_matmul(x, np.ones((3, 5)))

    def test_quantize_absmax_rejects_complex_and_underflow(self):
        for invalid in (np.array([1+2j]), [np.nextafter(0., 1.), 0.], [], [float('inf')]):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                quantize_absmax(invalid)


if __name__ == '__main__':
    unittest.main()
