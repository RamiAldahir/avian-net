import requests
from pathlib import Path
from time import sleep
from dotenv import load_dotenv
import os

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

load_dotenv()
API_KEY = os.getenv("XENO_KEY")

# Script lives in: dataset/
BASE_DIR = Path(__file__).resolve().parent

# Save here: dataset/recordings/
OUTPUT_DIR = BASE_DIR / "recordings"

COMMERCIAL_ONLY = True
MAX_RECORDINGS = 2
REQUEST_DELAY = 0.2

COMMERCIAL_LICENSES = {
    "//creativecommons.org/licenses/by/4.0/",
    "//creativecommons.org/licenses/by/3.0/",
    "//creativecommons.org/licenses/by-sa/4.0/",
    "//creativecommons.org/licenses/by-sa/3.0/",
}


# ----------------------------------------------------

def load_species(filename="species.txt"):
    path = BASE_DIR / filename
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def search_species(species):
    url = "https://xeno-canto.org/api/3/recordings"

    params = {
        "query": f'sp:"{species}" cnt:"United Kingdom"',
        "key": API_KEY,
        "page": 1
    }

    recordings = []

    while True:
        print(f"[{species}] page {params['page']}")

        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        print("Total returned:", len(data["recordings"]))

        recordings.extend(data["recordings"])

        if params["page"] >= int(data["numPages"]):
            break

        params["page"] += 1
        sleep(REQUEST_DELAY)

    return recordings


def acceptable(record):
    if record.get("q") not in ("A", "B", "C", "D"):
        return False

    if not COMMERCIAL_ONLY:
        return True

    lic = (record.get("lic") or "").lower()

    return (
        "by/4.0" in lic
        or "by-sa/4.0" in lic
        or "by/3.0" in lic
        or "by-sa/3.0" in lic
    )


def download(record, folder: Path):
    xcid = record["id"]

    filename = folder / f"XC{xcid}.mp3"

    if filename.exists():
        return

    raw_url = record["file"]

    if raw_url.startswith("//"):
        url = "https:" + raw_url
    elif raw_url.startswith("http"):
        url = raw_url
    else:
        url = "https://xeno-canto.org" + raw_url

    print("Downloading", url)

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with open(filename, "wb") as f:
        f.write(r.content)

    sleep(REQUEST_DELAY)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    species_list = load_species()

    for species in species_list:

        print("=" * 60)
        print(species)

        folder = OUTPUT_DIR / species.lower().replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)

        records = search_species(species)
        records = [r for r in records if acceptable(r)]

        print(f"{len(records)} usable recordings")

        for record in records[:MAX_RECORDINGS]:
            download(record, folder)


if __name__ == "__main__":
    species_list = load_species()
    print("SPECIES LOADED:", species_list)
    main()