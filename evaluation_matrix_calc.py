import json
import numpy as np
import geobleu
import os
import sys
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, required=True)
args = parser.parse_args()

with open(args.json_path, "r") as f:
    data = json.load(f)


#---- prepare the format ----
gen_list = []
ref_list = []
start_uid = 80000

for i in range(len(data['generated'])):
    uid = start_uid + i
    for step in data['generated'][i]:
        gen_list.append((uid, *step))
    for step in data['reference'][i]:
        ref_list.append((uid, *step))

#---- calculate the matrix ----

geobleu_score_2023 = geobleu.calc_geobleu_bulk_2023(gen_list, ref_list)
geobleu_score_2025 = geobleu.calc_geobleu_bulk(gen_list, ref_list)

print(f"GEO-BLEU 2023: {geobleu_score_2023:.4f}")
print(f"GEO-BLEU 2025: {geobleu_score_2025:.4f}")

dtw_score_2023 = geobleu.calc_dtw_bulk_2023(gen_list, ref_list)
dtw_score_2025 = geobleu.calc_dtw_bulk(gen_list, ref_list)

print(f"DTW 2023: {dtw_score_2023:.4f}")
print(f"DTW 2025: {dtw_score_2025:.4f}")

#---- save to run log ----
log_file = "evaluation_runlog.txt"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
job_name = os.getenv("PBS_JOBNAME", "Local_Run")
job_id = os.getenv("PBS_JOBID", "N/A")

log_entry = f"""
{"="*40}
Timestamp: {timestamp}
Katana Job: {job_name} (ID: {job_id})
File: {args.json_path}
----------------------------------------
GEO-BLEU 2023: {geobleu_score_2023:.4f}
GEO-BLEU 2025: {geobleu_score_2025:.4f}
DTW 2023: {dtw_score_2023:.4f}
DTW 2025: {dtw_score_2025:.4f}

"""
# Append to the log file so you don't overwrite previous results
with open(log_file, "a") as f:
    f.write(log_entry)

print(f"Results successfully saved to {log_file}")
