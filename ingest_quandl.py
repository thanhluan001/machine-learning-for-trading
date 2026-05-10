## Bug fix for bundle ingest quandl from AI

import numpy as np
import bcolz.utils

# Monkeypatch bcolz.utils.to_ndarray to handle dtype=None for empty arrays
_orig_to_ndarray = bcolz.utils.to_ndarray

def patched_to_ndarray(array, dtype, arrlen=None, safe=True):
    if dtype is None and type(array) == np.ndarray and len(array.strides) and array.strides[0] == 0:
        # If it's an empty array and dtype is None, let's infer it
        dtype = array.dtype
    return _orig_to_ndarray(array, dtype, arrlen, safe)

bcolz.utils.to_ndarray = patched_to_ndarray

from zipline.data import bundles
import os

bundle_name = 'quandl'
environ = os.environ

print(f"Starting ingestion for bundle '{bundle_name}'...")
bundles.ingest(bundle_name, environ, show_progress=True)
print("Ingestion completed successfully.")
