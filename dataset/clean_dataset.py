from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent

FOLDERS_TO_CLEAR = [
    BASE_DIR / "recordings",
    BASE_DIR / "processed",
]


def clear_folder(folder: Path):
    if not folder.exists():
        print(f"Skipping (does not exist): {folder}")
        return

    print(f"Clearing: {folder}")

    for item in folder.iterdir():

        if item.is_dir():
            shutil.rmtree(item)

        else:
            item.unlink()

    print("Done")


def main():

    print("This will delete all files inside:")

    for folder in FOLDERS_TO_CLEAR:
        print(" -", folder)

    confirm = input("\nContinue? (yes/no): ")

    if confirm.lower() != "yes":
        print("Cancelled")
        return

    for folder in FOLDERS_TO_CLEAR:
        clear_folder(folder)

    print("\nDataset folders cleaned.")


if __name__ == "__main__":
    main()