import pysam
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import scipy.stats
import os
import subprocess


def get_contig_indices(header_dict_loc):
    ids_dict = {}
    contigs_set = set()

    for header_el in header_dict_loc:
        el_name = header_el['SN']
        el_length = header_el['LN']
        contigs_set.add(el_name)

    contigs_names_list = np.array(list(contigs_set))
    argsort = np.argsort(contigs_names_list)

    for i, numb in enumerate(argsort):
        curr_el_name = contigs_names_list[numb]
        ids_dict[curr_el_name] = i

    return ids_dict


def report_reads_statistics(inp_filename,
                            stop_after_N_elements=-1,
                            plot_hist=True,
                            ):
    tic = time.time()
    bam_loc = pysam.AlignmentFile(inp_filename, "rb")
    header_dict_loc = bam_loc.header.to_dict()['SQ']
    ids_dict = get_contig_indices(header_dict_loc)
    n_elements = len(ids_dict)

    print("Total number of elements: ", n_elements)

    reads_per_element = np.zeros(n_elements)
    number_reads_categories = {"total": 0,
                               "passed": 0,
                               "qcfail": 0,
                               "unmapped": 0,
                               "multimapped": 0,
                               "insertions": 0,
                               "deletions": 0,
                               "skipped_regions": 0}

    el_counter = 0

    for header_el in header_dict_loc:
        el_name = header_el['SN']
        el_length = header_el['LN']
        el_number = ids_dict[el_name]

        reads_per_element[el_number] = bam_loc.count(el_name, read_callback='all')

        el_counter += 1
        if stop_after_N_elements > 0:
            if el_counter > stop_after_N_elements:
                break

    toc = time.time()
    print("Calculated in: %d" % (toc - tic))

    if plot_hist:
        plt.hist(reads_per_element, bins=50)
        plt.title("Number of reads per element")
        plt.show()


def calculate_cov_by_nt_deletions_on(bam_loc, el_name, el_length, min_mapq):
    cov = bam_loc.count_coverage(contig=el_name,
                                 quality_threshold=min_mapq,
                                 read_callback='all')
    a_cov, c_cov, g_cov, t_cov = cov
    a_cov = np.array(a_cov)
    c_cov = np.array(c_cov)
    g_cov = np.array(g_cov)
    t_cov = np.array(t_cov)
    return a_cov, c_cov, g_cov, t_cov


def count_nt_for_bam_deletions_on(inp_filename,
                                  min_mapq=30,
                                  expected_length=230,
                                  stop_after_N_elements=-1
                                  ):
    tic = time.time()
    bam_loc = pysam.AlignmentFile(inp_filename, "rb")
    header_dict_loc = bam_loc.header.to_dict()['SQ']
    ids_dict = get_contig_indices(header_dict_loc)
    n_elements = len(ids_dict)

    nt_counts_loc = np.zeros((n_elements, expected_length, 4))

    el_counter = 0

    for header_el in header_dict_loc:
        el_name = header_el['SN']
        el_length = header_el['LN']
        el_number = ids_dict[el_name]

        a_cov, c_cov, g_cov, t_cov = calculate_cov_by_nt_deletions_on(bam_loc, el_name, el_length, min_mapq)

        nt_counts_loc[el_number, 0: el_length, 0] = a_cov
        nt_counts_loc[el_number, 0: el_length, 1] = c_cov
        nt_counts_loc[el_number, 0: el_length, 2] = g_cov
        nt_counts_loc[el_number, 0: el_length, 3] = t_cov

        el_counter += 1
        if stop_after_N_elements > 0:
            if el_counter > stop_after_N_elements:
                break

    toc = time.time()
    print("Calculated in: %d" % (toc - tic))

    return nt_counts_loc


def encode_reference_sequences(seq_dict_loc,
                               nt_to_num_loc,
                               ad_5_size, ad_3_size,
                               expected_length = 230):
    encoding_loc = np.zeros((len(seq_dict_loc), expected_length), dtype=np.int)
    for i, seq_name in enumerate(sorted(list(seq_dict_loc.keys()))):
        for k, nt in enumerate(seq_dict_loc[seq_name]):
            encoding_loc[i,k] = nt_to_num_loc[nt]
        if len(seq_dict_loc[seq_name]) < expected_length:
            seq_length = len(seq_dict_loc[seq_name])
            length_difference = expected_length - len(seq_dict_loc[seq_name])
            #for k in range(len(seq_dict_loc[seq_name]), expected_length):
            encoding_loc[i, expected_length - ad_3_size : expected_length] = encoding_loc[i, seq_length - ad_3_size : seq_length]
            encoding_loc[i, seq_length - ad_3_size : expected_length - ad_3_size] = -1
    return encoding_loc


def trim_one_array(inp_array, beg, end):
    if len(inp_array.shape) < 2:
        print("Error! too few dimensions")
        sys.exit(1)
    elif len(inp_array.shape) == 2:
        return inp_array[:, beg: end]
    elif len(inp_array.shape) == 3:
        return inp_array[:, beg: end, :]
    else:
        print("Error! too many dimensions")
        sys.exit(1)


def trim_counts_dict(inp_dict, beg, end):
    out_dict = {}
    for sn in inp_dict:
        out_dict[sn] = trim_one_array(inp_dict[sn], beg, end)
    return out_dict


def identify_common_subs_dels(ref_vec_loc, count_dict_loc,
                              quantile_to_clip=0.9,
                              upper_lim_hist=100,
                              coverage_threshold=20,
                              ):
    sample_names = sorted(list(count_dict_loc.keys()))

    all_deletions = np.zeros((ref_vec_loc.shape[0], ref_vec_loc.shape[1], len(count_dict_loc)),
                             dtype=np.bool)
    tr_mean_coverage = np.zeros((ref_vec_loc.shape[0], len(count_dict_loc)), dtype=np.float32)
    low_coverage_transcripts_mask = np.zeros((ref_vec_loc.shape[0], ref_vec_loc.shape[1], len(count_dict_loc)),
                                             dtype=np.bool)

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30,10))

    for i, sn in enumerate(sample_names):
        current_counts_vec = count_dict_loc[sn]

        # find deletions and substitutions
        ref_seq_from_data = np.argmax(current_counts_vec, axis=2)
        is_there_sub = np.invert(
            ref_seq_from_data == ref_vec_loc)  # if letter in data doesn't correspond to the reference
        is_there_sub[np.sum(current_counts_vec, axis=2) == 0] = 1  # or if there no letter in the sata at all
        is_there_sub[ref_vec_loc == -1] = 0  # if the reference sequence there isn't present, it's not a deletion
        all_deletions[:, :, i] = is_there_sub

        # calculate coverage
        coverage_per_position = current_counts_vec.sum(axis=2)
        mean_coverage_per_transcript = np.mean(coverage_per_position, axis=1)
        tr_mean_coverage[:, i] = mean_coverage_per_transcript
        low_coverage_transcripts_mask[:, :, i] = (mean_coverage_per_transcript < coverage_threshold)[:, np.newaxis]

    # what are the mutations present everywhere or in individual sequences
    # the transcripts with low coverage are treated as unknown and therefore ignored
    all_deletions_unknown_yes = all_deletions.copy()
    all_deletions_unknown_yes[low_coverage_transcripts_mask] = 1
    # to take all the positions where all of the samples with good coverage show deletion
    dels_common = all_deletions_unknown_yes.all(axis=2)
    # to remove all the positions where coverage is too low in all the samples
    all_positions_too_low = low_coverage_transcripts_mask.all(axis=2)
    dels_common = dels_common & np.invert(all_positions_too_low)

    all_deletions_unknown_no = all_deletions.copy()
    all_deletions_unknown_no[low_coverage_transcripts_mask] = 0
    dels_any = all_deletions_unknown_no.any(axis=2)
    dels_not_common = dels_any & np.invert(dels_common)

    # calculate mean transcript coverage for each deletion among both common and not-common ones
    tr_mean_coverage_broadcasted = tr_mean_coverage[:, np.newaxis, :]
    tr_mean_coverage_broadcasted = np.broadcast_to(tr_mean_coverage_broadcasted, all_deletions.shape)  # broadcast it
    tr_mean_coverage_broadcasted = tr_mean_coverage_broadcasted.copy()
    # tr_mean_coverage_broadcasted[low_coverage_transcripts_mask] = 0 # make coverage zero for the low coverage transcripts
    # tr_mean_coverage_max_per_sample = np.max(tr_mean_coverage_broadcasted, axis=2)
    dels_common_3d = all_deletions & dels_common[:, :, np.newaxis]
    dels_not_common_3d = all_deletions & dels_not_common[:, :, np.newaxis]
    common_dels_mean_coverage = tr_mean_coverage_broadcasted[dels_common_3d]
    not_common_dels_mean_coverage = tr_mean_coverage_broadcasted[dels_not_common_3d]
    # for the deletions present in all samples, take the maximal coverage they occur at
    # common_dels_mean_coverage = tr_mean_coverage_max_per_sample[dels_common]

    #     common_dels_mean_coverage_values = common_dels_mean_coverage.flatten()
    #     common_dels_mean_coverage_values = common_dels_mean_coverage_values[common_dels_mean_coverage_values > 0]
    #     common_dels_mean_coverage_values = np.clip(common_dels_mean_coverage_values,
    #                                                a_min = np.min(common_dels_mean_coverage_values),
    #                                                a_max = np.quantile(common_dels_mean_coverage_values, quantile_to_clip))
    #     not_common_dels_mean_coverage_values = not_common_dels_mean_coverage.flatten()
    #     not_common_dels_mean_coverage_values = not_common_dels_mean_coverage_values[not_common_dels_mean_coverage_values > 0]
    #     not_common_dels_mean_coverage_values = np.clip(not_common_dels_mean_coverage_values,
    #                                                a_min = np.min(not_common_dels_mean_coverage_values),
    #                                                a_max = np.quantile(not_common_dels_mean_coverage_values, quantile_to_clip))

    #     fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30,10))

    #     axs[0,0].hist(common_dels_mean_coverage_values, bins=50, color='green')
    #     axs[0,0].set_title("Deletions present in all the samples", fontsize=24)
    #     axs[0,0].set_xlabel("Fragment coverage bin", fontsize=18)
    #     axs[0,0].set_ylabel("Number of deletions", fontsize=18)

    #     axs[0,1].hist(not_common_dels_mean_coverage_values, bins=50, color='red')
    #     axs[0,1].set_title("Deletions present in some samples only", fontsize=24)
    #     axs[0,1].set_xlabel("Fragment coverage bin", fontsize=18)
    #     axs[0,1].set_ylabel("Number of deletions", fontsize=18)

    #     axs[1,0].hist(common_dels_mean_coverage_values, bins=50, color='green', range=(0,upper_lim_hist))
    #     axs[1,0].set_title("Deletions present in all the samples", fontsize=24)
    #     axs[1,0].set_xlabel("Fragment coverage bin", fontsize=18)
    #     axs[1,0].set_ylabel("Number of deletions", fontsize=18)

    #     axs[1,1].hist(not_common_dels_mean_coverage_values, bins=50, color='red', range=(0,upper_lim_hist))
    #     axs[1,1].set_title("Deletions present in some samples only", fontsize=24)
    #     axs[1,1].set_xlabel("Fragment coverage bin", fontsize=18)
    #     axs[1,1].set_ylabel("Number of deletions", fontsize=18)

    #     plt.show()

    print("Number of common deletions: ", dels_common.sum())
    print("Number of non-common deletions: ", dels_not_common.sum())

    return dels_common, dels_not_common, tr_mean_coverage


def get_intervals_list_from_pairs(d_x, d_y):
    segments_loc = []
    curr_tr = -1
    start_pos = -1
    end_pos = -1
    curr_pos = -1
    for i, k in zip(d_x, d_y):
        if i != curr_tr:
            if curr_tr < 0:
                curr_tr = i
                start_pos = k
            else:
                segments_loc.append((curr_tr, start_pos, curr_pos + 1))
                curr_tr = i
                start_pos = k
        else:
            if k != curr_pos + 1:
                segments_loc.append((curr_tr, start_pos, curr_pos + 1))
                curr_tr = i
                start_pos = k
        curr_pos = k
    segments_loc.append((curr_tr, start_pos, curr_pos + 1))
    return segments_loc


def interval_to_seq(array, num_to_nt_loc):
    letters_list = [""] * array.shape[0]
    for i, value in enumerate(array):
        letters_list[i] = num_to_nt_loc[value]
    return "".join(letters_list)


def interval_to_seq_from_data(array, num_to_nt_loc):
    letters_list = [""] * array.shape[0]
    argmax_array = np.argmax(array, axis = 1)
    coverage_array = np.sum(array, axis = 1)
    for i, value, cov in zip(np.arange(array.shape[0]), argmax_array, coverage_array):
        if cov == 0:
            letters_list[i] = '-'
        else:
            letters_list[i] = num_to_nt_loc[value]
    return "".join(letters_list)


def one_hot_encode_3d(inp_labels):
    unique_categories = np.unique(inp_labels)
    n_categories = unique_categories.shape[0]
    one_hot_array = np.zeros((inp_labels.shape[0], n_categories))
    axis_0_index = np.arange(inp_labels.shape[0])
    axis_1_index = inp_labels
    one_hot_array[axis_0_index, axis_1_index] = 1
    return one_hot_array


def reference_to_one_hot(ref_2d):
    x_index = np.arange(ref_2d.shape[0]).repeat(ref_2d.shape[1])
    y_index = np.tile(np.arange(ref_2d.shape[1]), ref_2d.shape[0])
    flatten_reference = ref_2d.reshape(1,-1).flatten()
    return x_index, y_index, flatten_reference


def calculate_raw_mut_freqs(ref_vec_loc, count_dict_loc):
    x_index, y_index, flatten_reference = reference_to_one_hot(ref_vec_loc)
    mutation_rate_dict_loc = {}
    mutation_rate_fractions_dict_loc = {}

    sample_names = sorted(list(count_dict_loc.keys()))
    for i, sn in enumerate(sample_names):
        current_counts_vec = count_dict_loc[sn]

        # calculate overall mutation rates - not separated by nucleotide
        coverage_per_nt = current_counts_vec.sum(axis=2)  # get current nucleotide counts
        counts_real_nt = np.zeros_like(current_counts_vec)  # choose only the counts for reference nucleotides
        counts_real_nt[x_index, y_index, flatten_reference] = \
            current_counts_vec[x_index, y_index, flatten_reference]
        counts_real_nt = counts_real_nt.sum(axis=2)  # make counts for reference nucleotides into 2d array
        coverage_per_nt_nonzeros = coverage_per_nt.copy()
        coverage_per_nt_nonzeros[coverage_per_nt_nonzeros == 0] = 1  # avoid division by zero
        correct_nt_rate = counts_real_nt / coverage_per_nt_nonzeros
        correct_nt_rate[correct_nt_rate == 0] = 1
        mutation_rate = 1 - correct_nt_rate
        mutation_rate[
            ref_vec_loc == -1] = 0  # make sure mutation rate is zero at the places where we don't have reference
        mutation_rate_dict_loc[sn] = mutation_rate

        # calculate fractions of mutation rates per nucleotide
        frac_mut_rate = current_counts_vec / coverage_per_nt_nonzeros[:, :,
                                             np.newaxis]  # calculate fractions per nucleotide
        frac_mut_rate[x_index, y_index, flatten_reference] = 0  # zero fractions for the reference nucleotides
        frac_new_sum = frac_mut_rate.sum(axis=2)  # re-normalize the fractions, avoiding division by zero
        frac_new_sum[frac_new_sum == 0] = 1
        mutation_rate_fractions = frac_mut_rate / frac_new_sum[:, :, np.newaxis]
        mutation_rate_fractions[(ref_vec_loc == -1), :] = 0
        mutation_rate_fractions_dict_loc[sn] = mutation_rate_fractions
    return mutation_rate_dict_loc, mutation_rate_fractions_dict_loc


def calculate_nt_ratios_per_sample(thr_array,
                                   ref_vec_loc,
                                   nt_to_num_loc):
    nts_list = sorted(list(nt_to_num_loc.keys()))
    AC_meidan_ratios = np.zeros(thr_array.shape[0])

    for i in range(thr_array.shape[0]):
        curr_thr_array = thr_array[i]
        number_of_modified_array = np.zeros((len(nts_list), thr_array.shape[1]), dtype=np.int)

        for nt in nts_list:
            nt_id = nt_to_num_loc[nt]
            curr_nt_mask = ref_vec_loc != nt_id
            curr_nt_array = curr_thr_array.copy()
            curr_nt_array[curr_nt_mask] = 0
            number_of_modified_per_nt = (curr_nt_array > 0).sum(axis=1)
            number_of_modified_array[nt_id, :] = number_of_modified_per_nt.astype(np.int)

        A_modification_counts = number_of_modified_array[nt_to_num_loc['A'], :]
        C_modification_counts = number_of_modified_array[nt_to_num_loc['C'], :]
        AC_modification_counts = (A_modification_counts + C_modification_counts)  # [np.newaxis, :]
        total_modification_counts = number_of_modified_array.sum(axis=0)
        AC_modification_counts = AC_modification_counts[total_modification_counts > 0]
        total_modification_counts = total_modification_counts[total_modification_counts > 0]
        AC_ratios = np.divide(AC_modification_counts, total_modification_counts)
        median_AC_ratio = np.median(AC_ratios)
        AC_meidan_ratios[i] = median_AC_ratio

    return AC_meidan_ratios


def calculate_median_number_of_modified_per_sample(thr_array):
    meidan_N_modified = np.zeros(thr_array.shape[0])

    for i in range(thr_array.shape[0]):
        curr_thr_array = thr_array[i]
        number_of_modified = (curr_thr_array > 0).sum(axis=1)
        median_number_curr = np.median(number_of_modified)
        meidan_N_modified[i] = median_number_curr

    return meidan_N_modified


def calculate_dms_c_ratios(thr, thr_array,
                           ref_vec_loc,
                           nt_to_num_loc,
                           sample_pairs):
    ratios_list = []

    for tup in sample_pairs:
        name, dms_id, c_id = tup
        dms_rates = thr_array[dms_id, :, :]
        c_rates = thr_array[c_id, :, :]
        dms_modified_number = (dms_rates > 0).sum(axis=1)
        c_modified_number = (c_rates > 0).sum(axis=1)
        c_modified_number = c_modified_number[dms_modified_number > 0]
        dms_modified_number = dms_modified_number[dms_modified_number > 0]
        ratio = c_modified_number / dms_modified_number
        ratio_median = np.median(ratio)
        str_to_print = "Pair %s; theshold %.3f" % (name, thr)
        str_to_print += "; median ratio of modified positions per element in c/dms sample is"
        str_to_print += " %.2f" % ratio_median
        ratios_list.append(ratio_median)

    return ratios_list


def apply_threshold(inp_array,
                    ref_vec_loc,
                    nt_to_num_loc,
                    sample_pairs,
                    thr):
    thr_array = inp_array.copy()
    thr_array[thr_array < thr] = 0
    ratios_list = calculate_dms_c_ratios(thr, thr_array,
                                         ref_vec_loc,
                                         nt_to_num_loc,
                                         sample_pairs)
    AC_meidan_ratios = calculate_nt_ratios_per_sample(thr_array,
                                                      ref_vec_loc,
                                                      nt_to_num_loc)
    meidan_N_modified = calculate_median_number_of_modified_per_sample(thr_array)
    return ratios_list, AC_meidan_ratios, meidan_N_modified





def clean_out_profiles(inp_array_unf,
                        ref_vec_loc,
                        nt_to_num_loc,
                      threshold):
    inp_array = inp_array_unf.copy()
    inp_array[inp_array < threshold] = 0
    T_id = nt_to_num_loc['T']
    G_id = nt_to_num_loc['G']
    T_mask = ref_vec_loc == T_id
    G_mask = ref_vec_loc == G_id
    inp_array[:, T_mask] = 0
    inp_array[:, G_mask] = 0
    return inp_array


def calculate_raw_mut_counts(ref_vec_loc, count_dict_loc):
    x_index, y_index, flatten_reference = reference_to_one_hot(ref_vec_loc)

    sample_names = sorted(list(count_dict_loc.keys()))
    dim2, dim3, _ = count_dict_loc[sample_names[0]].shape
    mut_counts_array_loc = np.zeros((len(sample_names), dim2, dim3))
    total_counts_array_loc = np.zeros((len(sample_names), dim2, dim3))

    sample_names = sorted(list(count_dict_loc.keys()))
    for i, sn in enumerate(sample_names):
        current_counts_vec = count_dict_loc[sn]

        # calculate overall mutation rates - not separated by nucleotide
        coverage_per_nt = current_counts_vec.sum(axis=2)  # get current nucleotide counts
        counts_real_nt = np.zeros_like(current_counts_vec)  # choose only the counts for reference nucleotides
        counts_real_nt[x_index, y_index, flatten_reference] = \
            current_counts_vec[x_index, y_index, flatten_reference]
        counts_real_nt = counts_real_nt.sum(axis=2)  # make counts for reference nucleotides into 2d array
        curr_mutation_counts = coverage_per_nt - counts_real_nt
        curr_mutation_counts[
            ref_vec_loc == -1] = 0  # make sure mutation rate is zero at the places where we don't have reference
        mut_counts_array_loc[i, :, :] = curr_mutation_counts
        total_counts_array_loc[i, :, :] = coverage_per_nt
    return mut_counts_array_loc, total_counts_array_loc


def combine_samples_dict_to_array(inp_dict):
    sample_names = sorted(list(inp_dict.keys()))
    error_str = "dimensionality of arrays in this dictionary is not appropriate for this function."
    error_str += " please, use other function"
    assert len(inp_dict[sample_names[0]].shape) == 2, error_str

    dim2, dim3 = inp_dict[sample_names[0]].shape
    out_array = np.zeros((len(sample_names), dim2, dim3))

    for i, sn in enumerate(sample_names):
        out_array[i, :, :] = inp_dict[sn]

    return out_array


def combine_samples_dict_to_array(inp_dict):
    sample_names = sorted(list(inp_dict.keys()))
    error_str = "dimensionality of arrays in this dictionary is not appropriate for this function."
    error_str += " please, use other function"
    assert len(inp_dict[sample_names[0]].shape) == 2, error_str

    dim2, dim3 = inp_dict[sample_names[0]].shape
    out_array = np.zeros((len(sample_names), dim2, dim3))

    for i, sn in enumerate(sample_names):
        out_array[i, :, :] = inp_dict[sn]

    return out_array


def count_reads_and_umis_per_element(inp_filename):
    read_counts_dict = {}
    umis_counts_dict = {}
    bam_loc = pysam.AlignmentFile(inp_filename, "rb")
    header_dict_loc = bam_loc.header.to_dict()['SQ']
    for header_el in header_dict_loc:
        el_name = header_el['SN']
        el_length = header_el['LN']
        if el_name not in read_counts_dict:
            read_counts_dict[el_name] = 0
        if el_name not in umis_counts_dict:
            umis_counts_dict[el_name] = {}

        for read in bam_loc.fetch(el_name):
            umi = read.qname.split('_')[-1]
            read_counts_dict[el_name] += 1
            if umi not in umis_counts_dict[el_name]:
                umis_counts_dict[el_name][umi] = 0
            umis_counts_dict[el_name][umi] += 1
    return read_counts_dict, umis_counts_dict


def count_reads_and_umis_per_element_in_all_files(inp_filenames_dict):
    read_counts_dict = {}
    umis_counts_dict = {}
    for sn in sorted(list(inp_filenames_dict)):
        print("Sample: ", sn)
        tic = time.time()
        c_read_counts_dict, c_umis_counts_dict = count_reads_and_umis_per_element(inp_filenames_dict[sn])
        read_counts_dict[sn] = c_read_counts_dict
        umis_counts_dict[sn] = c_umis_counts_dict
        toc = time.time()
        print("counted in %d" % (toc - tic))
    return read_counts_dict, umis_counts_dict


def calculate_median_modifications_per_transcript(mut_counts_array_loc,
                                   ref_vec_loc,
                                   nt_to_num_loc):
    nts_list = sorted(list(nt_to_num_loc.keys()))
    median_modifications_per_nt_loc = np.zeros((mut_counts_array_loc.shape[0],
                                                len(nts_list),
                                                mut_counts_array_loc.shape[1]))

    for i in range(mut_counts_array_loc.shape[0]):
        for nt in nts_list:
            nt_id = nt_to_num_loc[nt]
            curr_nt_mask = ref_vec_loc != nt_id
            curr_nt_array = mut_counts_array_loc[i, :, :].copy()
            curr_nt_array_masked = np.ma.array(curr_nt_array, mask=curr_nt_mask)
            median_mut_count = np.ma.median(curr_nt_array_masked, axis=1)
            median_modifications_per_nt_loc[i, nt_id, :] = median_mut_count.astype(np.int)

    return median_modifications_per_nt_loc


def convert_bam_to_sam(inp_file, out_file):
    downsample_args = ['samtools', 'view', '-h']
    with open(inp_file, "r") as rf:
        with open(out_file, "w") as wf:
            subprocess.run(args=downsample_args, stdin=rf, stdout=wf, text=True)


def convert_bam_to_sam_folder(inp_folder, out_folder):
    for fn in os.listdir(inp_folder):
        if not fn.endswith('.bam'):
            continue
        inp_filename = os.path.join(inp_folder, fn)
        out_filename = os.path.join(out_folder, fn.replace('.bam', '.sam'))

        convert_bam_to_sam(inp_filename, out_filename)