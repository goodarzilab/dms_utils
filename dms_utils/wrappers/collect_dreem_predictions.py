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
    parser.add_argument("--output_file", help="", type=str)
    parser.add_argument("--conv_cutoff", help="", type=int)
    parser.add_argument("--num_base", help="surrounding bases for folding", type=int)
    parser.set_defaults(
        input_folder = '/rumi/shams/khorms/projects/SNIP_switchers/published_DMSseq_data/tomezsko_2020/stemsAC/StemA_C/downsampling/06_04_20_dreem',
        output_file = '/rumi/shams/khorms/projects/SNIP_switchers/published_DMSseq_data/tomezsko_2020/stemsAC/StemA_C/downsampling/collected_info/stemAC_06_04_20_dreem_cownsampling.csv',
        conv_cutoff=1,
        num_base=10,
    )
    args = parser.parse_args()
    return args


def parse_log_file(log_filename):
    with open(log_filename, 'r') as rf:
        for line in rf:
            if not line.startswith('Predicted number of clusters:'):
                continue
            k_clusters_str = line.replace("Predicted number of clusters: ", "")
            k_clusters = int(k_clusters_str)
    return k_clusters


# def get_best_likelihood(likelihood_file):
#     with open(likelihood_file, 'r') as rf:
#         lines = [x for x in rf.read().split('\n') if x != '']
#     return float(lines[-1])

def get_dot_notation_from_dot_file(dot_file):
    with open(dot_file, 'r') as rf:
        lines = [x for x in rf.read().split('\n') if x != '']
    sequence = lines[1]
    structure = lines[2]
    return sequence, structure


def get_number_of_filtered_reads(bit_vectors_filename):
    with open(bit_vectors_filename, 'r') as rf:
        lines = [x for x in rf.read().split('\n') if x != '']
    total_reads_used = int(lines[0].replace("Number of bit vectors used: ",""))
    unique_reads_used = int(lines[1].replace("Number of unique bit vectors used: ", ""))
    total_reads_discarded = int(lines[2].replace("Number of bit vectors discarded: ",""))
    return total_reads_used, unique_reads_used, total_reads_discarded

def get_cluster_fraction(prop_df, cluster_number):
    cluster_fraction_list = prop_df[prop_df['Cluster'] == cluster_number][' Obs Pi'].values.tolist()
    assert len(cluster_fraction_list) == 1
    cluster_fraction = float(cluster_fraction_list[0])
    return cluster_fraction


def get_proportions(proportions_file):
    prop_df = pd.read_csv(proportions_file, sep=',')
    cluster_1_fraction = get_cluster_fraction(prop_df, cluster_number = 1)
    cluster_2_fraction = get_cluster_fraction(prop_df, cluster_number = 2)
    return cluster_1_fraction, cluster_2_fraction


def get_convergence_iteration(likelihoods_file, conv_cutoff):
    with open(likelihoods_file, 'r') as rf:
        likelihoods = [float(x) for x in rf.read().split('\n') if x != '']
    i = 1
    not_converged = True
    while not_converged and i < len(likelihoods):
        diff = likelihoods[i] - likelihoods[i-1]
        if diff < conv_cutoff:
            not_converged = False
        i += 1
    return i


def get_bic_and_likelihood(log_likelihoods_bic_file):
    curr_df = pd.read_csv(log_likelihoods_bic_file, sep='\t')
    best_run_index_list = [x for x in curr_df['Run'] if x.endswith('-best')]
    assert len(best_run_index_list) == 1
    best_run_index = best_run_index_list[0]
    best_run_df = curr_df[curr_df['Run'] == best_run_index]
    assert best_run_df.shape[0] == 1
    likelihood = float(best_run_df['Log_likelihood'].values.tolist()[0])
    bic = float(best_run_df['BIC_score'].values.tolist()[0])
    return likelihood, bic


def collect_info_from_files(inp_folder, conv_cutoff, num_base):
    clustering_dict = {}
    for subf in os.listdir(inp_folder):
        full_subf_path = os.path.join(inp_folder, subf)
        if not os.path.isdir(full_subf_path):
            continue
        sample_name = subf
        if sample_name not in clustering_dict:
            clustering_dict[sample_name] = {}
        log_filename = os.path.join(full_subf_path, "log.txt")
        if not os.path.isfile(log_filename):
            continue
        k_clusters = parse_log_file(log_filename)
        clustering_dict[sample_name]['k_clusters'] = k_clusters

        bit_vectors_filename = os.path.join(full_subf_path, "BitVectors_Filter.txt")
        total_reads_used, unique_reads_used, total_reads_discarded = \
            get_number_of_filtered_reads(bit_vectors_filename)
        clustering_dict[sample_name]['total_reads_used'] = total_reads_used
        clustering_dict[sample_name]['unique_reads_used'] = unique_reads_used
        clustering_dict[sample_name]['total_reads_discarded'] = total_reads_discarded


        for K_subf in os.listdir(full_subf_path):
            K_subf_path = os.path.join(full_subf_path, K_subf)
            if not os.path.isdir(K_subf_path):
                continue
            best_run_subf = [x for x in os.listdir(K_subf_path) if x.endswith('-best')]
            assert len(best_run_subf) == 1
            best_run_subf = best_run_subf[0]
            best_run_folder = os.path.join(K_subf_path, best_run_subf)

            likelihoods_filename = os.path.join(best_run_folder, "Log_Likelihoods.txt")
            log_likelihoods_bic_file = os.path.join(K_subf_path, "log_likelihoods.txt")
            convergence_iter = get_convergence_iteration(likelihoods_filename, conv_cutoff)
            clustering_dict[sample_name]['%s_convergence_iter' % K_subf] = convergence_iter
            likelihood, bic = get_bic_and_likelihood(log_likelihoods_bic_file)
            clustering_dict[sample_name]["%s_likelihood" % (K_subf)] = likelihood
            clustering_dict[sample_name]["%s_bic" % (K_subf)] = bic

            if K_subf == 'K_1':
                cluster_1_dot_filename = os.path.join(best_run_folder, "%s-K1_Cluster1_expUp_%d_expDown_%d.dot" %
                                                      (sample_name, num_base, num_base))
                sequence, structure_K2_c1 = get_dot_notation_from_dot_file(cluster_1_dot_filename)
                clustering_dict[sample_name]['K1_cluster_structure'] = structure_K2_c1
                clustering_dict[sample_name]['sequence'] = sequence

            elif K_subf == 'K_2':
                cluster_1_dot_filename = os.path.join(best_run_folder, "%s-K2_Cluster1_expUp_%d_expDown_%d.dot" %
                                                      (sample_name, num_base, num_base))
                cluster_2_dot_filename = os.path.join(best_run_folder, "%s-K2_Cluster2_expUp_%d_expDown_%d.dot" %
                                                      (sample_name, num_base, num_base))
                _, structure_K2_c1 = get_dot_notation_from_dot_file(cluster_1_dot_filename)
                _, structure_K2_c2 = get_dot_notation_from_dot_file(cluster_2_dot_filename)
                clustering_dict[sample_name]['K2_cluster_1_structure'] = structure_K2_c1
                clustering_dict[sample_name]['K2_cluster_2_structure'] = structure_K2_c2
                proportions_filename = os.path.join(best_run_folder, "Proportions.txt")
                fraction_1, fraction_2 = get_proportions(proportions_filename)
                clustering_dict[sample_name]['fraction_cluster_1'] = fraction_1
                clustering_dict[sample_name]['fraction_cluster_2'] = fraction_2

            else:
                print("Error! Weird K subfolder")
                sys.exit(1)

    clustering_df_loc = pd.DataFrame.from_dict(clustering_dict, orient='index')
    return clustering_df_loc


def main():
    args = handler()
    clustering_df_loc = collect_info_from_files(args.input_folder, args.conv_cutoff, args.num_base)
    clustering_df_loc.to_csv(args.output_file, sep='\t')


if __name__ == '__main__':
    main()