---

## SETUP INSTRUCTIONS

### 1. Clone the repository

git clone https://github.com/RamiAldahir/avian-net
cd avian-net

### 2. Create and activate the virtual environment

Windows:
--------------------
python -m venv .venv
.venv\Scripts\Activate.ps1

Windows (cmd):
--------------------
python -m venv .venv
.venv\Scripts\activate.bat

macOS / Linux:
--------------------
python3 -m venv .venv
source .venv/bin/activate

### 3. Install dependancies

pip install -r requirements.txt


## DOWNLOADING THE DATASET

To download bird audio recordings from the Xeno-Canto API:
python dataset/download_xeno.py
This script will automatically fetch and save recordings locally for use in training or analysis.

You can modify this script to fetch more recordings for each bird species by modifiying the:
MAX_RECORDINGS = <>
variable

Bird species to gather are listed in the file: dataset/species.txt


### IMPORTANT ###

Create a .env file and add your API key in there:

XENO_KEY = "your-cool-xeno-key"