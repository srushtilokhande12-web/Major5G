import glob
import os
import pandas as pd

DATA_DIR = os.path.join("5G-production-dataset", "5G-production-dataset")

def load_all(data_dir=DATA_DIR):
    files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    files = [f for f in files if "MACOSX" not in f
             and not os.path.basename(f).startswith("._")]
    files.sort()

    frames = []
    for f in files:
        parts = os.path.relpath(f, data_dir).split(os.sep)
        # Streaming: App/Mobility/Content/file.csv
        # Download:  Download/Mobility/file.csv
        app = parts[0]
        mobility = parts[1]
        content = parts[2] if len(parts) == 4 else "none"

        df = pd.read_csv(f)
        df["trace_id"] = os.path.splitext(parts[-1])[0]
        df["app"] = app
        df["mobility"] = mobility
        df["content"] = content

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce"
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    data = load_all()
    print("Rows:", len(data))
    print("Traces:", data["trace_id"].nunique())
    print("Bad timestamps:", data["Timestamp"].isna().sum())
    print(data.groupby(["app", "mobility"])["trace_id"].nunique())
    data.to_pickle("merged_raw.pkl")