import sys
sys.path.append('/rumi/shams/khorms/programs/dms_utils')
import os
import argparse
import multiprocessing
import pandas as pd
import numpy as np
from statistics import median as med

import dms_utils.utils.dreem_utils as dreem_utils
import dms_utils.utils.dreem_original_functions as dof


def handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="", type=str)
    parser.add_argument("--n_processes", help="", type=int)

    parser.add_argument("--reference_fasta_filename", help="", type=str)
    parser.add_argument("--num_runs", help="", type=int)
    parser.add_argument("--num_base", help="surrounding bases for folding", type=int)
    parser.add_argument("--norm_bases", help="number of top bases per 100 nt to clip", type=int)

    parser.set_defaults(
        input_folder = '/rumi/shams/khorms/projects/SNIP_switchers/published_DMSseq_data/tomezsko_2020/stemsAC/StemA_C/downsampling/06_04_20_dreem',
        n_processes = 15,

        reference_fasta_filename="/rumi/shams/khorms/projects/SNIP_switchers/published_DMSseq_data/tomezsko_2020/stemsAC/StemA_C/reference/MRPS21_200_nt.fa",
        num_runs = 10,
        num_base = 0,
        norm_bases = 10,
    )
    args = parser.parse_args()
    return args

def declare_global_variables(args):
    global reference_fasta_filename
    global num_runs
    global num_base
    global norm_bases

    reference_fasta_filename = args.reference_fasta_filename
    num_runs = args.num_runs
    num_base = args.num_base
    norm_bases = args.norm_bases


def ConstraintFoldDraw(ref_filename, clustMuFile, expUp, expDown, norm_bases):
    sample = clustMuFile.split('/')[-4]
    file_dir = os.path.dirname(clustMuFile)

    # Create trimmed reference fasta file
    clustMuFileContents = open(clustMuFile)
    first_line = clustMuFileContents.readline().strip()
    second_line = clustMuFileContents.readline().strip()
    clustMuFileContents.close()
    first_line_split = first_line.strip().split()
    ref_info = first_line_split[1]
    ref_file, ref = ref_info.split(';')[0], ref_info.split(';')[1]
    #ref_file = input_dir + ref_file + '.fasta'
    #refs_seq = Parse_FastaFile(ref_file)
    refs_seq = dof.Parse_FastaFile(ref_filename)
    entire_seq = refs_seq[ref]
    second_line_split = second_line.strip().split()
    indices = second_line_split[1].split(':')[0]
    start, end = int(indices.split(',')[0]), int(indices.split(',')[1])
    mod_start = max(1, start - expUp)
    mod_end = min(end + expDown, len(entire_seq))
    trim_seq = entire_seq[mod_start - 1:mod_end]
    trimref_filename = file_dir + '/' + ref + '_trimUp_' + str(expUp) + \
        '_trimDown_' + str(expDown) + '.fa'
    ref_name = ref + '_' + str(mod_start) + '_' + str(mod_end)
    dof.Create_FastaFile(trimref_filename, ref_name, trim_seq)

    # Gather mus for every k and normalize them
    clusts_mus = pd.read_csv(clustMuFile, sep='\t', skiprows=2,
                             index_col=False)
    rows, K = len(clusts_mus), len(clusts_mus.columns) - 1
    norm_clusts_mus = np.zeros((rows, K))
    for k in range(K):
        mus = clusts_mus['Cluster_' + str(k + 1)]
        norm_value = med(np.sort(mus)[-1:-(norm_bases+1):-1])  # Median of mus
        norm_mus = mus / norm_value  # Normalize the mus
        norm_mus[norm_mus > 1.0] = 1.0  # Cap at 1
        norm_clusts_mus[:, k] = norm_mus
    norm_clusts_mus = np.around(norm_clusts_mus, decimals=3)

    # Drawing of structure for each k

    print("Current file: ")
    print(clustMuFile)

    for k in range(K):
        clust_name = file_dir + '/' + sample + '-K' + str(K) + '_Cluster' + str(k+1)

        const_filename = clust_name + '_expUp_' + str(expUp) + '_expDown_' + \
            str(expDown) + '_const.txt'
        const_file = open(const_filename, 'w')
        for i in range(len(clusts_mus)):
            pos = clusts_mus['Position'][i]
            mod_pos = pos - mod_start + 1  # Pos wrt trimmed seq
            mu = str(norm_clusts_mus[i][k])
            if mu == 'nan':  # Happens in UT
                mu = '0'
            if entire_seq[pos-1] == 'T' or entire_seq[pos-1] == 'G':
                mu = '-999'
            if mod_pos > 0 and mod_start <= pos <= mod_end:  # Can be < 0 because its wrt trimmed seq
                const_file.write(str(mod_pos) + '\t' + mu + '\n')
        const_file.close()

        # Folding using RNAstructure
        ct_filename = clust_name + '_expUp_' + str(expUp) + '_expDown_' + \
            str(expDown) + '.ct'
        dot_filename = clust_name + '_expUp_' + str(expUp) + '_expDown_' + \
            str(expDown) + '.dot'
        pic_filename = clust_name + '_basesExpanded_' + str(expUp) + '.ps'
        fold_command = '/rumi/shams/khorms/programs/RNAstructure/exe/Fold -m 3 ' + trimref_filename + ' -dms ' + \
            const_filename + ' ' + ct_filename
        ct2dot_command = '/rumi/shams/khorms/programs/RNAstructure/exe/ct2dot ' + ct_filename + ' ALL ' + \
            dot_filename
        # draw_command = '/rumi/shams/khorms/programs/RNAstructure/exe/draw ' + dot_filename + ' ' + \
        #     pic_filename + ' -S ' + const_filename
        add_database_command = "export DATAPATH=/rumi/shams/khorms/programs/RNAstructure/data_tables"
        full_command = "%s ; %s ; %s " % (add_database_command, fold_command,
                                              ct2dot_command)
        os.system(full_command)
        # os.system(fold_command)
        # os.system(ct2dot_command)
        # os.system(draw_command)

        # Delete unnecessary files
        # os.system('rm ' + const_filename)
        # os.system('rm ' + ct_filename)
        # os.system('rm ' + dot_filename)

    os.system('rm ' + trimref_filename)

def launch_folding(inp_filename):
    ConstraintFoldDraw(reference_fasta_filename, inp_filename, num_base, num_base, norm_bases)


def make_list_filenames(inp_folder):
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


def launch_processes(n_processes, inp_folder):
    pool = multiprocessing.Pool(n_processes)
    list_of_filenames = make_list_filenames(inp_folder)
    print(list_of_filenames)
    pool.map(launch_folding, list_of_filenames)


def main():
    args = handler()
    declare_global_variables(args)
    launch_processes(args.n_processes, args.input_folder)


if __name__ == '__main__':
    main()