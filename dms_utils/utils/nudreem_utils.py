from . import glob_vars
from . import io
import pandas as pd
import numpy as np
from scipy.stats import hypergeom
import itertools
import copy
import sys
import networkx as nx
import os
import pickle


def convert_bitstring_to_numpy(inp_string):
    out_array = np.zeros(len(inp_string), dtype=np.int8)
    for i in range(len(inp_string)):
        out_array[i] = glob_vars.bitstring_letters_to_num[inp_string[i]]
    return out_array


def convert_bitstring_to_numpy_for_df(inp_df, column_name):
    dim_axis_0 = inp_df.shape[0]
    dim_axis_1 = len(inp_df.iloc[0][column_name])
    out_array = np.zeros((dim_axis_0, dim_axis_1), dtype=np.int8)
    inp_df.index = np.arange(inp_df.shape[0])
    for index, row in inp_df.iterrows():
        out_array[index, :] = convert_bitstring_to_numpy(row[column_name])
    # arrays_series = inp_df.apply(lambda x: convert_bitstring_to_numpy(x[column_name]),
    #                             axis = 1)
    return out_array


def get_bitstring_coverage(inp_array):
    out_array = np.zeros_like(inp_array)
    out_array[inp_array != -1] = 1
    return out_array.sum(axis=0)


def make_mask_by_coverage(coverage_array, coverage_threshold = 20):
    return coverage_array >= coverage_threshold


def get_bitstring_mutation_fraction(inp_array):
    coverage_vector = get_bitstring_coverage(inp_array)
    out_array = np.zeros_like(inp_array)
    out_array[inp_array == 1] = 1
    mutation_rates = out_array.sum(axis=0) / coverage_vector
    return mutation_rates


def make_mask_mutation_rates(mutation_rates,
                             mut_rate_lower = 0.005,
                             mut_rate_upper = 0.25):
    return np.logical_and(mutation_rates >= mut_rate_lower,
                          mutation_rates <= mut_rate_upper)


def make_TG_mask_from_string(ref_sequence_string,
                             masked_out_nts = {'T','G'}):
    out_mask = np.zeros(len(ref_sequence_string), dtype=np.bool)
    for i in range(len(ref_sequence_string)):
        if ref_sequence_string[i] not in masked_out_nts:
            out_mask[i] = 1
    return out_mask


def combine_several_masks(masks_list):
    if len(masks_list) == 1:
        return masks_list[0]
    out_mask = masks_list[0]
    for next_mask in masks_list[1:]:
        out_mask = np.logical_and(out_mask, next_mask)
    return out_mask


def count_mutation_co_occurence(inp_array,
                               weights = None):
    if weights is None:
        weights = np.ones(inp_array.shape[0], dtype=np.bool)
    inp_array_t = inp_array.transpose()
    pairwise_and_loc = np.logical_and(inp_array_t[np.newaxis, :, :], inp_array_t[:, np.newaxis, :])
    weighted_pairwise_and_loc = pairwise_and_loc * weights[np.newaxis, np.newaxis, :]
    weighted_sum = weighted_pairwise_and_loc.sum(axis=2)
    weighted_sum_upper = np.triu(weighted_sum, k = 1)
    return weighted_sum_upper


def reshape_co_occurence_df_to_pairwise(inp_df):
    stacked_df = inp_df.stack()
    stacked_df = stacked_df[stacked_df != 0].rename_axis(('source', 'target')).reset_index(name='weight')
    return stacked_df


def get_enrichment_score(cooccurence_matrix_loc,
                         mutation_counts_loc,
                         total_reads_number_loc):
    enr_score_matrix = np.zeros_like(cooccurence_matrix_loc, dtype = np.float)
    positions_number = cooccurence_matrix_loc.shape[0]
    for i, k in itertools.combinations(np.arange(positions_number), 2):
        cooccurence_value = cooccurence_matrix_loc[i, k]
        mut_count_i = mutation_counts_loc[i]
        mut_count_k = mutation_counts_loc[k]
        # get the expected intersection
        expected = hypergeom.mean(M = total_reads_number_loc,
                                  n = mut_count_i,
                                  N = mut_count_k)
        if cooccurence_value > expected:
            sign = 1
            pvalue = hypergeom.sf(k = cooccurence_value,
                                  M = total_reads_number_loc,
                                  n = mut_count_i,
                                  N = mut_count_k)
        else:
            sign = -1
            pvalue = hypergeom.cdf(k = cooccurence_value,
                                  M = total_reads_number_loc,
                                  n = mut_count_i,
                                  N = mut_count_k)
        log_pvalue = -1 * np.log(pvalue)
        signed_log_pvalue = sign * log_pvalue
        if np.isinf(signed_log_pvalue) or np.isnan(signed_log_pvalue):
            signed_log_pvalue = 0
        enr_score_matrix[i, k] = signed_log_pvalue
    return enr_score_matrix


def get_expected_co_occurence(cooccurence_matrix_loc,
                         mutation_counts_loc,
                         total_reads_number_loc):
    expected_matrix = np.zeros_like(cooccurence_matrix_loc, dtype = np.float)
    positions_number = cooccurence_matrix_loc.shape[0]
    for i, k in itertools.combinations(np.arange(positions_number), 2):
        mut_count_i = mutation_counts_loc[i]
        mut_count_k = mutation_counts_loc[k]
        # get the expected intersection
        expected = hypergeom.mean(M = total_reads_number_loc,
                                  n = mut_count_i,
                                  N = mut_count_k)
        expected_matrix[i, k] = expected
    return expected_matrix


def zero_adjacent_nucleotides_interactions(inter_matrix, labels, min_distance = 3):
    out_matrix = inter_matrix.copy()
    for i, k in itertools.combinations(np.arange(labels.shape[0]), 2):
        if abs(labels[i] - labels[k]) <= min_distance:
            out_matrix[i, k] = 0
    return out_matrix


def calculate_mutation_thresholds(inp_df,
                                  higher_fraction = 0.1,
                                  do_print = True):
    lengths = inp_df.apply(lambda x: len(x['Bit_vector']), axis = 1)
    unique_lengths = list(set(lengths))
    assert len(unique_lengths) == 1
    length = unique_lengths[0]
    higher_limit = int(round(higher_fraction * length))
    n_muts = inp_df['N_Mutations']
    mad = abs(n_muts - n_muts.median()).median()
    nmuts_thresh = n_muts.median() + (3 * mad / 0.6745)
    if do_print:
        print("median + 3 * std: %.2f" % nmuts_thresh)
        print("%.2f of the length: %.2f" % (higher_fraction, higher_limit))
    nmuts_thresh = max(int(round(nmuts_thresh)),  higher_limit)
    return nmuts_thresh


def filter_1_number_of_mutations(row, filters_names_loc):
    passed = True
    if row['N_Mutations'] > filters_names_loc['nmuts_thresh']:
        passed = False
    return passed


def filter_2_fraction_of_informative_bits(row, filters_names_loc):
    passed = True
    bit_string = row['Bit_vector']
    if (bit_string.count('.') + bit_string.count('?') +
           bit_string.count('N')) >= filters_names_loc['info_thresh'] * len(bit_string):
        passed = False
    return passed


def filter_3_distance_between_mutations(row, filters_names_loc):
    passed = True
    bit_string = row['Bit_vector']
    latest_mutbit_index = -1000
    for i in range(len(bit_string)):
        if bit_string[i] == '1':
            if i - latest_mutbit_index < 4:
                passed = False
            latest_mutbit_index = i
    return passed


def filter_4_bits_surrounding_mutations(row, filters_names_loc):
    passed = True
    bit_string = row['Bit_vector']
    invalid_set = ['.1', '?1', '1.', '1?']
    for i in range(len(bit_string)):
        if bit_string[i:i + 2] in invalid_set:
            passed = False
    return passed


def dreem_style_read_filters(inp_df,
                             filters_list_loc,
                             filters_names_loc,
                             parameters_dict_loc,
                             do_print = True):
    if do_print:
        print("Total number of reads: %d" % (inp_df.shape[0]))
    passing_all_filters = np.ones(inp_df.shape[0], dtype = np.bool)
    for filter_func, filter_name in zip(filters_list_loc, filters_names_loc):
        curr_filtered_array = inp_df.apply(lambda x: filter_func(x, parameters_dict_loc),
                                          axis = 1)
        reads_passed = inp_df.shape[0] - curr_filtered_array.sum()
        fraction = reads_passed / inp_df.shape[0]
        string_to_write = "Number of reads filtered out by filter %s: " % (filter_name)
        string_to_write += "%d (fraction %.2f)" % (reads_passed, fraction)
        if do_print:
            print(string_to_write)
        passing_all_filters = np.logical_and(passing_all_filters, curr_filtered_array.to_numpy())
    total_reads_filtered = inp_df.shape[0] - passing_all_filters.sum()
    fraction = total_reads_filtered / inp_df.shape[0]
    string_to_write = "Number of reads filtered out by at least one filter: "
    string_to_write += "%d (fraction %.2f)" % (total_reads_filtered, fraction)
    string_to_write += "\n number of the ones left: %d" % (inp_df.shape[0] - total_reads_filtered)
    if do_print:
        print(string_to_write)
    return passing_all_filters


def count_occurence_symbol_per_bit_string(inp_df,
                                          column_name,
                                          symbol):
    return inp_df.apply(lambda x: x[column_name].count(symbol),
                       axis = 1)


def get_positions_from_mask(inp_mask):
    return np.arange(inp_mask.shape[0])[inp_mask]


def bitvector_to_numpy_pipeline(inp_filename,
                                info_thresh = 0.2):
    set_of_filters = [filter_1_number_of_mutations,
                      filter_2_fraction_of_informative_bits,
                      filter_3_distance_between_mutations,
                      filter_4_bits_surrounding_mutations]
    set_of_filter_names = ["filter_1_number_of_mutations",
                           "filter_2_fraction_of_informative_bits",
                           "filter_3_distance_between_mutations",
                           "filter_4_bits_surrounding_mutations"]

    bitvect_df = io.read_bitvector_to_df(inp_filename)
    bitvect_reference_string = io.read_bitvector_reference_sequence(inp_filename)
    nmuts_thresh = calculate_mutation_thresholds(bitvect_df)
    filter_parameters_dict = {"nmuts_thresh": nmuts_thresh,
                                 'info_thresh': info_thresh}  # got value from a hard coded value in Run_DREEM.py script
    read_filters_mask = dreem_style_read_filters(bitvect_df,
                                          set_of_filters,
                                          set_of_filter_names,
                                          filter_parameters_dict)
    bitvect_df_filtered = bitvect_df[read_filters_mask]
    bitvect_array = convert_bitstring_to_numpy_for_df(bitvect_df_filtered,
                                                    column_name="Bit_vector")
    return bitvect_array, bitvect_reference_string


def bitvector_array_to_graph_pipeline(bitvect_array,
                                      bitvect_reference_string,
                                      coverage_threshold = 100,
                                      mut_rate_lower = 0.005,
                                      mut_rate_upper = 0.25
                                      ):
    bitvect_coverage = get_bitstring_coverage(bitvect_array)
    bitvect_mut_rates = get_bitstring_mutation_fraction(bitvect_array)
    coverage_mask = make_mask_by_coverage(bitvect_coverage,
                                          coverage_threshold = coverage_threshold)
    mut_rate_mask = make_mask_mutation_rates(bitvect_mut_rates,
                                            mut_rate_lower = mut_rate_lower,
                                            mut_rate_upper = mut_rate_upper)
    TG_mask = make_TG_mask_from_string(bitvect_reference_string)
    all_masks_combined = combine_several_masks([coverage_mask,
                                                 mut_rate_mask,
                                                 TG_mask])
    bitvect_array_muts_only = np.zeros_like(bitvect_array)
    bitvect_array_muts_only[bitvect_array == 1] = 1
    bitvect_variable_muts_only_array = bitvect_array_muts_only[:, all_masks_combined]
    bitvect_unique_muts_only_array, bitvect_unique_muts_only_counts = np.unique(bitvect_variable_muts_only_array,
                                     axis = 0,
                                     return_counts = True)
    # cooccurence_nw = count_mutation_co_occurence(bitvect_unique_muts_only_array)
    cooccurence_weighted = count_mutation_co_occurence(bitvect_unique_muts_only_array,
                                                                        weights = bitvect_unique_muts_only_counts)
    total_reads_number = bitvect_array.shape[0]
    mutation_counts_array = bitvect_variable_muts_only_array.sum(axis = 0)
    enrichm_score_matrix = get_enrichment_score(cooccurence_weighted,
                                     mutation_counts_array,
                                     total_reads_number)
    position_numbers = get_positions_from_mask(all_masks_combined)
    enrichm_score_matrix_no_adj = zero_adjacent_nucleotides_interactions(enrichm_score_matrix,
                                                                        labels = position_numbers,
                                                                        min_distance = 3)
    enrichm_score_matrix_no_adj_tanh = np.tanh(enrichm_score_matrix_no_adj)
    weighted_scores_tanh_df = pd.DataFrame(data = enrichm_score_matrix_no_adj_tanh,
                                             index = position_numbers,
                                             columns = position_numbers)
    weighted_scores_tanh_df_pairwise = reshape_co_occurence_df_to_pairwise(weighted_scores_tanh_df)
    G_weighted_scores_tanh = nx.from_pandas_edgelist(weighted_scores_tanh_df_pairwise,
                                            edge_attr=True)
    return G_weighted_scores_tanh, all_masks_combined


def positive_weights_subgraph(inp_graph):
    out_graph = copy.deepcopy(inp_graph)
    edge_attributes = nx.get_edge_attributes(out_graph, 'weight')
    edges_to_remove = set()
    for edge in out_graph.edges():
        if edge_attributes[edge] <= 0:
            edges_to_remove.add(edge)
    out_graph.remove_edges_from(edges_to_remove)
    return out_graph


def rename_graph_for_karate(inp_graph):
    assert nx.is_connected(inp_graph), "The graph must be connected! Otherwise karate won't work on it"
    return nx.convert_node_labels_to_integers(inp_graph,
                                       ordering = 'default',
                                       label_attribute = 'position')


def fill_in_node_attritute_dict_for_unknowns(inp_graph, inp_dict,
                                             missing_value = 0):
    all_nodes = inp_graph.nodes()
    out_dict = {}
    for node in all_nodes:
        if node in inp_dict:
            out_dict[node] = inp_dict[node]
        else:
            out_dict[node] = missing_value
    return out_dict


def std_calculation_through_mad(inp_array):
    mad = np.median(np.abs(inp_array - np.median(inp_array)))
    std = mad / 0.6745
    return std


def get_cooccurence_weighted(bitvect_array):
    bitvect_array_muts_only = np.zeros_like(bitvect_array)
    bitvect_array_muts_only[bitvect_array == 1] = 1
    bitvect_unique_muts_only_array, bitvect_unique_muts_only_counts = np.unique(bitvect_array_muts_only,
                                     axis = 0,
                                     return_counts = True)
    cooccurence_weighted = count_mutation_co_occurence(bitvect_unique_muts_only_array,
                                                        weights = bitvect_unique_muts_only_counts)
    total_reads_number = bitvect_array.shape[0]
    mutation_counts_array = bitvect_array_muts_only.sum(axis=0)
    return cooccurence_weighted, mutation_counts_array, total_reads_number



def filtered_bit_vector_to_graph(bitvect_array):
    cooccurence_weighted, mutation_counts_array, total_reads_number = get_cooccurence_weighted(bitvect_array)
    enrichm_score_matrix = get_enrichment_score(cooccurence_weighted,
                                     mutation_counts_array,
                                     total_reads_number)
    position_numbers = np.arange(bitvect_array.shape[1])
    enrichm_score_matrix_no_adj = zero_adjacent_nucleotides_interactions(enrichm_score_matrix,
                                                                        labels = position_numbers,
                                                                        min_distance = 3)
    enrichm_score_matrix_no_adj_tanh = np.tanh(enrichm_score_matrix_no_adj)
    weighted_scores_tanh_df = pd.DataFrame(data = enrichm_score_matrix_no_adj_tanh,
                                             index = position_numbers,
                                             columns = position_numbers)
    weighted_scores_tanh_df_pairwise = reshape_co_occurence_df_to_pairwise(weighted_scores_tanh_df)
    G_weighted_scores_tanh = nx.from_pandas_edgelist(weighted_scores_tanh_df_pairwise,
                                            edge_attr=True)
    return G_weighted_scores_tanh


def filtered_bit_vector_to_graph_observed_expected(bitvect_array):
    cooccurence_weighted, mutation_counts_array, total_reads_number = get_cooccurence_weighted(bitvect_array)



    enrichm_score_matrix = get_enrichment_score(cooccurence_weighted,
                                     mutation_counts_array,
                                     total_reads_number)
    position_numbers = np.arange(bitvect_array.shape[1])
    enrichm_score_matrix_no_adj = zero_adjacent_nucleotides_interactions(enrichm_score_matrix,
                                                                        labels = position_numbers,
                                                                        min_distance = 3)
    enrichm_score_matrix_no_adj_tanh = np.tanh(enrichm_score_matrix_no_adj)
    weighted_scores_tanh_df = pd.DataFrame(data = enrichm_score_matrix_no_adj_tanh,
                                             index = position_numbers,
                                             columns = position_numbers)
    weighted_scores_tanh_df_pairwise = reshape_co_occurence_df_to_pairwise(weighted_scores_tanh_df)
    G_weighted_scores_tanh = nx.from_pandas_edgelist(weighted_scores_tanh_df_pairwise,
                                            edge_attr=True)
    return G_weighted_scores_tanh



def relabel_graph_by_original_structure(graph,
                                        structure_1_index, structure_2_index,
                                        structure_1_label = 1, structure_2_label = 2,
                                        attribute_label = 'sw'):
    assert structure_1_index.shape[0] == structure_2_index.shape[0]
    assert np.logical_and(structure_1_index, structure_2_index).sum() == 0
    assert np.logical_or(structure_1_index, structure_2_index).sum() == structure_1_index.shape[0]

    labels_dict = {}
    for i in range(structure_1_index.shape[0]):
        if structure_1_index[i]:
            labels_dict[i] = structure_1_label
        elif structure_2_index[i]:
            labels_dict[i] = structure_2_label
        else:
            sys.exit(1)
    nx.set_node_attributes(graph,
                           labels_dict,
                           name=attribute_label)
    return graph


def get_communities_by_attribute(graph, attribute):
    attributes_dict = nx.get_node_attributes(graph, attribute)
    comms = []
    for value in set(attributes_dict.values()):
        curr_comm = []
        for k in attributes_dict:
            if attributes_dict[k] == value:
                curr_comm.append(k)
        comms.append(curr_comm)
    return comms


def erdos_renyi_modularity(graph, communities):
    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    q = 0

    for community in communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        nc = c.number_of_nodes()
        q += mc - (m * nc * (nc - 1)) / (n * (n - 1))
    if q == 0:
        score = None
    else:
        score = (1 / m) * q
    return score


def pad_shape_profiles_to_reference_sequence(inp_folder,
                                             out_folder,
                                             sequence_length,
                                             subseq_coordinates,
                                             fill_unknown_with = -999,
                                             old_suffix = "_shape_profile.pickle",
                                             new_suffix = "_shape_profile_padded.pickle"):
    for short_filename in os.listdir(inp_folder):
        if not short_filename.endswith(old_suffix):
            continue
        input_filename = os.path.join(inp_folder, short_filename)
        output_filename = os.path.join(out_folder, short_filename.replace(old_suffix, new_suffix))
        curr_shape_profile_dict = pickle.load(open(input_filename, 'rb'))
        assert len(curr_shape_profile_dict) == 1
        curr_profile = curr_shape_profile_dict[list(curr_shape_profile_dict.keys())[0]]
        full_profile = np.full(sequence_length, fill_unknown_with, dtype=np.float)
        full_profile[subseq_coordinates[0] : subseq_coordinates[1]] = curr_profile
        new_dict = {list(curr_shape_profile_dict.keys())[0] : full_profile}
        pickle.dump(new_dict, open(output_filename, 'wb'))