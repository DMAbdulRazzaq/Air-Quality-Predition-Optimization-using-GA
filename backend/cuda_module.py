from numba import cuda
import numpy as np
from sklearn.metrics import mean_squared_error

@cuda.jit
def compute_rmse_gpu(y_true, y_pred, errors):
    idx = cuda.grid(1)
    if idx < y_true.size:
        diff = y_true[idx] - y_pred[idx]
        errors[idx] = diff * diff

gpu_warning_shown = False

def gpu_rmse(y_true, y_pred):
    global gpu_warning_shown

    try:
        if not cuda.is_available():
            raise Exception("CUDA not available")

        n = len(y_true)
        errors = np.zeros(n, dtype=np.float32)

        d_y_true = cuda.to_device(y_true)
        d_y_pred = cuda.to_device(y_pred)
        d_errors = cuda.to_device(errors)

        threads = 256
        blocks = (n + threads - 1) // threads

        compute_rmse_gpu[blocks, threads](d_y_true, d_y_pred, d_errors)

        errors = d_errors.copy_to_host()
        return np.sqrt(np.mean(errors))

    except Exception:
        if not gpu_warning_shown:
            print("⚠️ CUDA not available → Falling back to CPU")
            gpu_warning_shown = True
        return np.sqrt(mean_squared_error(y_true, y_pred))