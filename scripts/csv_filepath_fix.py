import pandas as pd

# Input and output file paths
input_file = "data/core50_tr.csv"
output_file = "data/core50_tr.csv"

# Read CSV
df = pd.read_csv(input_file, dtype=str)  # read everything as string to avoid issues

# Remove '/data5/quang/' from all cells
df = df.applymap(lambda x: x.replace('/data5/quang/', '') if isinstance(x, str) else x)

# Save to new CSV
df.to_csv(output_file, index=False)
