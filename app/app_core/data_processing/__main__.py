import os
from sys import argv, exit
from .XMLparser import XMLparser
from .text_extract import ManualFeatureExtract


# Check whether input dir is valid
if len(argv) < 2 or (not os.path.isdir(argv[1])):
    print("[ERROR] Input dir not found!")
    exit(1)

input_dir = argv[1]

# Read Output path and make directory
if len(argv) > 2:
    outpath = argv[2]
else:
    outpath = "Dataset"

# Force re-processing of existing files or not
force = len(argv) > 3 and bool(argv[3])

if not os.path.isdir(outpath):
    os.makedirs(outpath)

# Parse xml files in the input dir
for fp in os.listdir(input_dir):
    full_path = os.path.join(input_dir, fp)
    if not (os.path.isdir(full_path) or full_path.endswith(".xml")):
        continue

    name = fp.split(".")[0] + ".csv"
    out_file = os.path.join(outpath, name)
    if force or (not os.path.isfile(out_file)):
        df = ManualFeatureExtract().process_df(XMLparser(full_path))
        df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"[INFO] Finished processing {full_path} file!")

print("[INFO] Processing done!")
exit(0)
