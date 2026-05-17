from pathlib import Path

# Define root directory
root = Path(r"C:\rag-toolkit\experiments")

# Define output file
output_file = root / "paths.txt"

# Collect all paths excluding __pycache__
paths = [
    str(path)
    for path in root.rglob("*")
    if "__pycache__" not in path.parts
]

# Write paths to txt file
output_file.write_text("\n".join(paths), encoding="utf-8")

print(f"Paths exported to: {output_file}")