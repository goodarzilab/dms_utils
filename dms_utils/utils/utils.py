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


def get_sample_to_filename_dict(inp_folder, inp_suffix):
    filenames = sorted([x for x in os.listdir(inp_folder) if x.endswith(inp_suffix)])
    sample_to_filename_dict = {x.replace(inp_suffix, ""):
                                         os.path.join(inp_folder, x) for x in filenames}
    return sample_to_filename_dict


def get_element_info(header_el, ids_dict):
    el_name = header_el['SN']
    el_length = header_el['LN']
    el_number = ids_dict[el_name]
    return el_name, el_length, el_number


def get_start_end_positions(el_length, adapter_5_size, adapter_3_size):
    start_poition = adapter_5_size
    end_position = el_length - adapter_3_size
    real_length = end_position - start_poition
    return start_poition, end_position


def read_callback_all(read):
    keep_the_read = True
    if (read.flag & (0x4 | 0x100 | 0x200 | 0x400)):
        keep_the_read = False
    return keep_the_read


def read_to_bit_array(read,
                      ref_start_poition, ref_end_position,
                      base_quality_threshold):
    reference_length = ref_end_position - ref_start_poition
    bitvector_array = np.full(reference_length, fill_value=-1, dtype=np.int8)
    aligned_pairs = read.get_aligned_pairs(with_seq=True)
    for tup in aligned_pairs:
        query_position, reference_position, reference_sequence = tup
        if reference_position is None:
            continue  # insertion in the read

        if reference_position < ref_start_poition or reference_position >= ref_end_position:
            continue  # outside of the region of interest

        coordinate_reference = reference_position - ref_start_poition

        if reference_sequence.isupper(): # if the letter is upper, it's a match
            bitvector_array[coordinate_reference] = 0
        else:
            if read.query_alignment_qualities[query_position - read.query_alignment_start] < base_quality_threshold:
                continue # check the base quality
            else:
                bitvector_array[coordinate_reference] = 1
    return bitvector_array


def compare_expected_observed_mutation_counts(bitvectors_array,
                                              count_coverage,
                                              accepted_disagreement_fraction,
                                              do_print):
    a_cov, c_cov, g_cov, t_cov = count_coverage
    coverage_array = np.zeros((4, len(a_cov)), dtype=np.int)
    coverage_array[0, :] = np.array(a_cov)
    coverage_array[1, :] = np.array(c_cov)
    coverage_array[2, :] = np.array(g_cov)
    coverage_array[3, :] = np.array(t_cov)
    max_values = coverage_array.max(axis=0)
    sum_coverage = coverage_array.sum(axis=0)
    mutations_expected = sum_coverage - max_values

    mutations_count_calculated = bitvectors_array.copy()
    mutations_count_calculated[mutations_count_calculated < 0] = 0
    mutations_count_calculated = mutations_count_calculated.sum(axis=0)

    is_count_the_same = mutations_expected != mutations_count_calculated
    calculated_correctly = True
    percentage_off_reference = (is_count_the_same.sum() / is_count_the_same.shape[0])
    if do_print:
        if percentage_off_reference != 0:
            print("Percentage of positions off of reference", percentage_off_reference)
            print("Observed")
            print(mutations_count_calculated[is_count_the_same])
            print("Expected")
            print(coverage_array[:, is_count_the_same])
    if (is_count_the_same.sum() / is_count_the_same.shape[0]) > accepted_disagreement_fraction:
        calculated_correctly = False
    return calculated_correctly




def reads_to_bitvector_arrays(inp_filename,
                              adapter_5_size, adapter_3_size,
                              stop_after_N_elements=-1,
                              stop_after_N_reads=-1,
                              base_quality_threshold=30,
                              accepted_disagreement_fraction = 0.1,
                              do_print = True,
                              how_often_print = 100,
                              ):
    bitvectors_out_dict = {}
    bam_loc = pysam.AlignmentFile(inp_filename, "rb")
    header_dict_loc = bam_loc.header.to_dict()['SQ']
    ids_dict = get_contig_indices(header_dict_loc)
    el_counter = 0
    tic = time.time()

    for header_el in header_dict_loc:
        el_name, el_length, el_number = get_element_info(header_el, ids_dict)
        start_poition, end_position = get_start_end_positions(el_length, adapter_5_size, adapter_3_size)
        real_length = end_position - start_poition

        curr_reads_count = bam_loc.count(el_name, read_callback='all')
        curr_bitvectors_array = np.zeros((curr_reads_count, real_length), dtype=np.int8)

        read_count = 0
        for read in bam_loc.fetch(el_name):
            if read_callback_all(read):
                bitvector_array = read_to_bit_array(read,
                                                    start_poition, end_position,
                                                    base_quality_threshold)
                curr_bitvectors_array[read_count, :] = bitvector_array

                read_count += 1
                if read_count == stop_after_N_reads:
                    break

        count_coverage = bam_loc.count_coverage(contig = el_name,
                                                          start = start_poition, stop = end_position,
                                                          quality_threshold = base_quality_threshold,
                                                          read_callback='all')
        assert read_count == curr_reads_count
        assert compare_expected_observed_mutation_counts(curr_bitvectors_array,
                                                        count_coverage,
                                                         accepted_disagreement_fraction,
                                                         do_print = do_print)
        bitvectors_out_dict[el_name] = curr_bitvectors_array

        el_counter += 1
        if stop_after_N_elements > 0:
            if el_counter > stop_after_N_elements:
                break
        if el_counter % how_often_print == 0:
            if do_print:
                toc = time.time()
                print("%d reference fragments calculated. spent %d seconds for the last %d" %
                      (el_counter, (toc - tic), how_often_print))
                tic = time.time()
    return bitvectors_out_dict


def get_mutations_per_read_from_bit_arrays(bit_arrays_dict):
    total_reads = 0

    for element in bit_arrays_dict:
        total_reads += bit_arrays_dict[element].shape[0]
    mutation_counts_array = np.zeros(total_reads, dtype=np.int32)

    current_index = 0
    for element in bit_arrays_dict:
        current_shift = bit_arrays_dict[element].shape[0]
        current_mutated_array = bit_arrays_dict[element].copy()
        current_mutated_array[current_mutated_array < 0] = 0
        mutation_counts_array[current_index : current_index + current_shift] = current_mutated_array.sum(axis=1)
        current_index += current_shift

    return mutation_counts_array

def get_non_inform_per_read_from_bit_arrays(bit_arrays_dict):
    total_reads = 0

    for element in bit_arrays_dict:
        total_reads += bit_arrays_dict[element].shape[0]
    fraction_non_inform_array = np.zeros(total_reads, dtype=np.float)

    current_index = 0
    for element in bit_arrays_dict:
        current_shift = bit_arrays_dict[element].shape[0]
        current_non_inform_array = bit_arrays_dict[element].copy()
        current_non_inform_array[current_non_inform_array != -1] = 0
        current_non_inform_array[current_non_inform_array == -1] = 1
        count_non_inform = current_non_inform_array.sum(axis=1)
        fraction_non_inform = count_non_inform /  current_non_inform_array.shape[1]
        fraction_non_inform_array[current_index : current_index + current_shift] = fraction_non_inform
        current_index += current_shift

    return fraction_non_inform_array


def calculate_number_mutations_threshold(all_reads_mutation_counts_array,
                                         number_of_stds = 3):
    median_mutations_number = np.median(all_reads_mutation_counts_array)
    std = np.std(all_reads_mutation_counts_array)
    curr_mut_thr = median_mutations_number + number_of_stds * std
    return curr_mut_thr


def filter_by_mutation_counts(current_bit_matrix,
                              n_mutations_threshold):
    current_mutated_array = current_bit_matrix.copy()
    current_mutated_array[current_mutated_array < 0] = 0
    mutation_counts = current_mutated_array.sum(axis=1)
    return mutation_counts > n_mutations_threshold


def filter_by_fraction_covered_positions(current_bit_matrix,
                                  max_fraction_non_inform):
    current_non_inform_array = current_bit_matrix.copy()
    current_non_inform_array[current_non_inform_array != -1] = 0
    current_non_inform_array[current_non_inform_array == -1] = 1
    count_non_inform = current_non_inform_array.sum(axis=1)
    fraction_non_inform = count_non_inform / current_non_inform_array.shape[1]
    return fraction_non_inform > max_fraction_non_inform


def filter_by_distance_between_mutations(current_bit_matrix,
                                         max_prohib_distance = 3):
    bool_mutations = current_bit_matrix == 1
    out_mask = np.zeros(current_bit_matrix.shape[0], dtype = np.bool)
    for i in range(1, max_prohib_distance + 1):
        left_subset = bool_mutations[:, :-i]
        right_subset = bool_mutations[:, i:]
        mut_of_the_right_distance = np.logical_and(left_subset, right_subset)
        curr_distance_mask = mut_of_the_right_distance.sum(axis=1) > 0
        out_mask = np.logical_or(out_mask, curr_distance_mask)
    return out_mask


def filter_by_positions_surrounding_mutations(current_bit_matrix):
    bool_mutations = current_bit_matrix == 1
    bool_non_covered = current_bit_matrix == -1
    mutations_left = bool_mutations[:,:-1]
    mutations_right = bool_mutations[:, 1:]
    non_covered_left = bool_non_covered[:,:-1]
    non_covered_right = bool_non_covered[:, 1:]
    mut_then_non_covered = np.logical_and(mutations_left, non_covered_right)
    mut_then_non_covered_mask = mut_then_non_covered.sum(axis=1) > 0
    non_covered_then_mut = np.logical_and(non_covered_left, mutations_right)
    non_covered_then_mut_mask = non_covered_then_mut.sum(axis=1) > 0
    return np.logical_or(mut_then_non_covered_mask, non_covered_then_mut_mask)


def make_filter_masks_based_on_filter_function(bit_arrays_dict,
                                               filter_function):
    masks_dict = {}
    for element in bit_arrays_dict:
        masks_dict[element] = filter_function(bit_arrays_dict[element])
        assert masks_dict[element].shape[0] == bit_arrays_dict[element].shape[0]
    return masks_dict


def apply_specified_filters_to_bit_arrays_dict(bit_arrays_dict,
                                               filters_list,
                                               filters_names_list):
    masks_dict = {}
    for filt, filt_name in zip(filters_list, filters_names_list):
        masks_dict[filt_name] = make_filter_masks_based_on_filter_function(
                                               bit_arrays_dict,
                                               filt)
    return masks_dict


def count_number_of_reads_filtered_by_mask(mask_dict):
    total_number_of_reads = 0
    total_number_of_reads_filtered_out = 0

    for element in mask_dict:
        total_number_of_reads += mask_dict[element].shape[0]
        total_number_of_reads_filtered_out += mask_dict[element].sum()

    return total_number_of_reads, total_number_of_reads_filtered_out


def get_fraction_filtered_reads(total_number_of_reads, total_number_of_reads_filtered_out):
    return total_number_of_reads_filtered_out /total_number_of_reads


def combine_mask_dicts(list_of_mask_dicts):
    length_of_mask_dict = len(list_of_mask_dicts[0])
    for i in range(len(list_of_mask_dicts)):
        assert len(list_of_mask_dicts[i]) == length_of_mask_dict
    combined_mask_dict = {}
    for mask_dict in list_of_mask_dicts:
        for element in mask_dict:
            if element not in combined_mask_dict:
                combined_mask_dict[element] = mask_dict[element]
            combined_mask_dict[element] = np.logical_or(combined_mask_dict[element],
                                                        mask_dict[element])
    return combined_mask_dict
