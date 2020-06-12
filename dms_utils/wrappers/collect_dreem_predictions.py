import sys
sys.path.append('/rumi/shams/khorms/programs/dms_utils')
import os
import argparse
import pandas as pd

import dms_utils.utils.dreem_utils as dreem_utils
import dms_utils.utils.dreem_original_functions as dof

def handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="", type=str)
    parser.set_defaults(
        input_folder = '/rumi/shams/khorms/projects/SNIP_switchers/published_DMSseq_data/tomezsko_2020/stemsAC/StemA_C/downsampling/06_04_20_dreem_copy',
    )
    args = parser.parse_args()
    return args


def collect_filenames(inp_folder, ):
    filenames_list = []
    for subf in os.listdir(inp_folder):
        full_subf_path = os.path.join(inp_folder, subf)
        if not os.path.isdir(full_subf_path):
            continue
        for K_subf in os.listdir(full_subf_path):
            K_subf_path = os.path.join(full_subf_path, K_subf)
            if not os.path.isdir(K_subf_path):
                continue
            best_run_subf = [x for x in os.listdir(K_subf_path) if x.endswith('-best')]
            assert len(best_run_subf) == 1
            best_run_subf = best_run_subf[0]
            curr_Clusters_Mu_filename = os.path.join(K_subf_path, best_run_subf, "Clusters_Mu.txt")
            filenames_list.append(curr_Clusters_Mu_filename)
    return filenames_list


def main():
    args = handler()


if __name__ == '__main__':
    main()