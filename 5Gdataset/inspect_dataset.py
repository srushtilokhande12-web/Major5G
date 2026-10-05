import pandas as pd
import glob
import os

# STEP 1: Find actual CSV files
data_dir = "5G-production-dataset"

all_files = glob.glob(
    os.path.join(data_dir, "**", "*.csv"),
    recursive=True
)

# Ignore macOS metadata files
all_files = [
    f for f in all_files
    if "MACOSX" not in f and not os.path.basename(f).startswith("._")
]

print("=" * 60)
print("STEP 1: DATASET INVENTORY")
print("=" * 60)

print(f"Total trace files found: {len(all_files)}")

print("\nFirst 10 files:")
for f in all_files[:10]:
    print(f)


# STEP 2: Load one real CSV
print("\n" + "=" * 60)
print("STEP 2: DATASET STRUCTURE")
print("=" * 60)

if len(all_files) == 0:
    print("ERROR: No CSV files found.")
    exit()

file = all_files[0]

print(f"\nInspecting file:\n{file}")

df = pd.read_csv(file, sep=None, engine="python")

print("\nShape:", df.shape)

print("\nColumns:")
print(list(df.columns))

print("\nData Types:")
print(df.dtypes)

print("\nFirst 10 rows:")
print(df.head(10))


# STEP 3: Data quality
print("\n" + "=" * 60)
print("STEP 3: DATA QUALITY")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# STEP 4: Summary statistics
print("\n" + "=" * 60)
print("STEP 4: SUMMARY STATISTICS")
print("=" * 60)

print(df.describe(include="all"))