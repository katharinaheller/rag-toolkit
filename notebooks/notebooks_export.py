import os

# Define source directory and output file
source_dir = r"C:\rag-toolkit\notebooks"
output_file = r"C:\rag-toolkit\all_files.txt"

# Open output file
with open(output_file, "w", encoding="utf-8") as out_file:
    # Walk through directory recursively
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # Write file header
            out_file.write(f"\n\n===== FILE: {file_path} =====\n\n")

            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    out_file.write(f.read())
            except Exception as e:
                # Handle binary or unreadable files
                out_file.write(f"[ERROR READING FILE: {e}]\n")

print("Done! All files written to:", output_file)