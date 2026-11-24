import tempfile
import unittest
from pathlib import Path

import numpy as np

from geograph2.embedding import (
    dequantize_tessera_int8,
    generate_synthetic_tessera_cube,
    load_tessera_npy_pair,
    load_tessera_zarr_compat,
    quantize_tessera_int8,
    save_tessera_zarr_compat,
)


class TestTesseraEmbeddingIO(unittest.TestCase):
    def test_zarr_compat_zip_roundtrip_int8_scale(self):
        cube, _, _ = generate_synthetic_tessera_cube(16, 16, dim=128, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            p = save_tessera_zarr_compat(cube, Path(tmp) / "tessera_tile.zarr.zip")
            loaded = load_tessera_zarr_compat(p)

        self.assertEqual(loaded.embeddings.shape, (16, 16, 128))
        self.assertEqual(loaded.version, "1.1")
        self.assertEqual(loaded.variant, "dclimate")
        # int8 quantization max error should be < 1/127 of max abs (~0.01)
        max_err = float(np.max(np.abs(loaded.embeddings - cube.embeddings)))
        self.assertLess(max_err, 0.015)

    def test_npy_pair_dequantization(self):
        cube, _, _ = generate_synthetic_tessera_cube(8, 8, dim=128, seed=11)
        q, s = quantize_tessera_int8(cube.embeddings)
        with tempfile.TemporaryDirectory() as tmp:
            qp = Path(tmp) / "grid.npy"
            sp = Path(tmp) / "grid_scales.npy"
            np.save(qp, q)
            np.save(sp, s)
            loaded = load_tessera_npy_pair(qp, sp)
        np.testing.assert_allclose(loaded.embeddings, dequantize_tessera_int8(q, s), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
