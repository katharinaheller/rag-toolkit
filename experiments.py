from pathlib import Path

# Root directory to scan
ROOT_DIR = Path(r"C:\rag-toolkit\rag\evaluation")

# Output file
OUTPUT_FILE = ROOT_DIR / "all_files_dump.txt"

# Directories to exclude completely
EXCLUDED_DIRS = {
    "__pycache__",
    ".history",
    ".venv",
    "venv",
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".gradle",
    ".vscode",
}

# File extensions to skip
EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".exe",
    ".dll",
    ".so",
    ".class",
    ".jar",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".pdf",
    ".mp4",
    ".mp3",
    ".wav",
}

# Maximum file size in bytes (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


def should_skip(path: Path) -> bool:
    # Exclude directories anywhere in the path
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return True

    # Exclude unwanted file extensions
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    return False


with OUTPUT_FILE.open("w", encoding="utf-8", errors="ignore") as out_file:
    for file_path in ROOT_DIR.rglob("*"):
        try:
            # Skip non-files
            if not file_path.is_file():
                continue

            # Skip excluded paths
            if should_skip(file_path):
                continue

            # Skip large files
            if file_path.stat().st_size > MAX_FILE_SIZE:
                continue

            # Read file content
            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            # Write absolute path and content
            out_file.write(f"{file_path.resolve()} :\n")
            out_file.write(content)
            out_file.write("\n")
            out_file.write("=" * 120)
            out_file.write("\n\n")

            print(f"Processed: {file_path}")

        except Exception as e:
            print(f"Skipped {file_path}: {e}")

print(f"\nDone. Output written to:\n{OUTPUT_FILE}")