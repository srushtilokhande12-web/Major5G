import numpy as np
import pandas as pd
from make_windows import make_windows
from make_features import BASE_FEATURES

feature_cols = BASE_FEATURES + ["dCQI", "dSNR", "CQI_rmean5", "CQI_rstd5",
                                "SNR_rmean5", "SNR_rstd5"]
data = pd.read_pickle("features.pkl")

for W in [5, 20]:
    for h in [1, 3, 5]:
        X, y, meta = make_windows(data, feature_cols, W, h)
        np.savez_compressed(f"windows_W{W}_h{h}.npz", X=X, y=y)
        meta.to_pickle(f"windows_W{W}_h{h}_meta.pkl")
        print(f"W={W} h={h}: X{X.shape}")