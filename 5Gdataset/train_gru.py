import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from split_data import get_data
from baselines import raw_cqi_window

SEED = 42
HORIZONS = [1, 3, 5]

class GRUNet(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):                 # x: (N, W, F)
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)   # predicts y - CQI(t)

def tensors(d, name, scaler):
    X, y, _ = d[name]
    base = raw_cqi_window(X, scaler)[:, -1]
    return (torch.tensor(X), torch.tensor(y - base, dtype=torch.float32),
            torch.tensor(base, dtype=torch.float32), torch.tensor(y))

if __name__ == "__main__":
    torch.manual_seed(SEED)
    split = json.load(open("split.json"))
    scaler = joblib.load("scaler.pkl")
    W = json.load(open("best_config.json"))["W"]

    for h in HORIZONS:
        d = get_data(W, h, split, scaler)
        Xtr, rtr, _, _ = tensors(d, "train", scaler)
        Xva, rva, bva, yva = tensors(d, "val", scaler)

        model = GRUNet(Xtr.shape[2])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()
        best, bad, best_state = 1e9, 0, None

        for epoch in range(60):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss_fn(model(Xtr[idx]), rtr[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred = (model(Xva) + bva).clamp(0, 15)
                mae = (pred - yva).abs().mean().item()
            if mae < best:
                best, bad = mae, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break

        print(f"h={h}: best val MAE={best:.4f} (stopped at epoch {epoch+1})")
        torch.save(best_state, f"models/gru_h{h}.pt")