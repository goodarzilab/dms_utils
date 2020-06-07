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


def read_fasta(infile):
    tr_dict_loc = {}
    with open(infile, 'r') as f:
        split_string = f.read().split('>')
        for entry in split_string:
            if entry == '':
                continue
            seq_start = entry.find('\n')
            annotation = entry[:seq_start]
            sequence = entry[seq_start + 1:].replace('\n', '')
            tr_dict_loc[annotation] = sequence

    return tr_dict_loc


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


def save_dict_of_np_arrays(inp_dict, out_folder):
    for sn in list(inp_dict.keys()):
        fn = os.path.join(out_folder, "%s.npy" % sn)
        np.save(fn, inp_dict[sn])


def load_dict_of_np_arrays(inp_folder):
    out_dict = {}
    for fn in os.listdir(inp_folder):
        if not fn.endswith('npy'):
            continue
        out_dict[fn.replace('npy','')] = np.load(os.path.join(inp_folder, fn))

    return out_dict


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


def plot_substitutions_along_transcript(ref_vec_loc, count_dict_loc):
    sample_names = sorted(list(count_dict_loc.keys()))
    subs_common = np.ones_like(ref_vec_loc).astype(np.bool)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
    for sn in sample_names:
        current_counts_vec = count_dict_loc[sn]
        ref_seq_from_data = np.argmax(current_counts_vec, axis = 2)
        is_there_sub = np.invert(ref_seq_from_data == ref_vec_loc)
        is_there_sub[np.sum(current_counts_vec, axis = 2) == 0] = 1
        is_there_sub[ref_vec_loc == -1] = 0
        subs_common = subs_common & is_there_sub
        axs[0].plot(np.arange(ref_vec_loc.shape[1]), is_there_sub.sum(axis=0), label=sn)
        axs[0].set_title("Number of substitutions / deletions along the transcript", fontsize=24)
    axs[0].set_xlabel("Coordinate along the length of a fragment", fontsize=18)
    axs[0].set_ylabel("Number of fragments with substitutions", fontsize=18)

    axs[0].legend()
    plt.show()


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


def print_subs_dels(ref_vec_loc, count_dict_loc,
                    dels_not_common,
                    tr_mean_coverage_loc,
                    num_to_nt_loc,
                    buffer=10,
                    min_coverage=20,
                    do_print_consensus=True):
    sample_names = sorted(list(count_dict_loc.keys()))
    d_x, d_y = np.where(dels_not_common == 1)
    segments_loc = get_intervals_list_from_pairs(d_x, d_y)

    for segment in segments_loc:
        el_id = segment[0]
        start = segment[1]
        end = segment[2]
        start_ext = max(0, start - buffer)
        end_ext = min(ref_vec_loc.shape[1], end + buffer)
        print("Element number %d; start %d, end %d" % (el_id, start, end))
        reference_sequence_before = interval_to_seq(ref_vec_loc[el_id][start_ext: start], num_to_nt_loc)
        reference_sequence = interval_to_seq(ref_vec_loc[el_id][start: end], num_to_nt_loc)
        reference_sequence_after = interval_to_seq(ref_vec_loc[el_id][end: end_ext], num_to_nt_loc)
        reference_sequence_string = "reference: %s - %s - %s" % (reference_sequence_before,
                                                                 reference_sequence,
                                                                 reference_sequence_after)
        print(reference_sequence_string)
        for i, sample in enumerate(sample_names):
            current_sequence_before = interval_to_seq_from_data(count_dict_loc[sample][el_id, start_ext: start, :],
                                                                num_to_nt_loc)
            current_sequence = interval_to_seq_from_data(count_dict_loc[sample][el_id, start: end, :], num_to_nt_loc)
            current_sequence_after = interval_to_seq_from_data(count_dict_loc[sample][el_id, end: end_ext, :],
                                                               num_to_nt_loc)
            coverage_value = tr_mean_coverage_loc[el_id, i]
            if coverage_value < min_coverage:
                continue
            current_sequence_string = "sample %d:  %s - %s - %s  ; coverage: %d" % (i,
                                                                                    current_sequence_before,
                                                                                    current_sequence,
                                                                                    current_sequence_after,
                                                                                    coverage_value)
            if do_print_consensus:
                print(current_sequence_string)
        print()
    print("Total segments: ", len(segments_loc))


def print_whole_element(el_id, ref_vec_loc, count_dict_loc,
                        tr_mean_coverage_loc,
                        num_to_nt_loc,
                        min_coverage=20,
                        do_print_consensus=True):
    sample_names = sorted(list(count_dict_loc.keys()))

    print("Element number %d" % (el_id))
    current_counts_vec = count_dict_loc[sn]
    reference_sequence = interval_to_seq(ref_vec_loc[el_id], num_to_nt_loc)
    reference_sequence_string = "reference: %s" % reference_sequence
    print(reference_sequence_string)
    for i, sample in enumerate(sample_names):
        current_sequence = interval_to_seq_from_data(count_dict_loc[sample][el_id, :, :], num_to_nt_loc)
        coverage_value = tr_mean_coverage_loc[el_id, i]
        if coverage_value < min_coverage:
            continue
        current_sequence_string = "sample %d:  %s  ; coverage: %d" % (i,
                                                                      current_sequence,
                                                                      coverage_value)
        if do_print_consensus:
            print(current_sequence_string)
    if do_print_consensus:
        print()

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


def plot_mut_rate_distributions_per_nt(mut_rate_loc, ref_vec_loc,
                                       tr_mean_coverage_loc, nt_to_num_loc,
                                       xlim=0.1, ylim=10000,
                                       min_coverage=20):
    modif_rates_per_nt_dict_loc = {}
    nts_loc = sorted(list(nt_to_num_loc.keys()))
    sample_names = sorted(list(mut_rate_loc.keys()))

    fig, axs = plt.subplots(nrows=len(sample_names), ncols=len(nts_loc))
    fig.set_size_inches(24, 32)

    for i, sn in enumerate(sample_names):
        current_rates_vec = mut_rate_loc[sn]
        current_rates_vec_nz = current_rates_vec != 0
        current_coverage = tr_mean_coverage_loc[:, i]
        good_enough_coverage = current_coverage >= min_coverage
        modif_rates_per_nt_dict_loc[sn] = {}
        for k, nt in enumerate(nts_loc):
            nt_value = nt_to_num_loc[nt]
            curr_nt_mask = ref_vec_loc == nt_value
            modified_nt_mask = (current_rates_vec_nz & curr_nt_mask & good_enough_coverage[:, np.newaxis])
            curr_modif_rates = current_rates_vec[modified_nt_mask]
            curr_modif_rates = curr_modif_rates[curr_modif_rates != 0]
            modif_rates_per_nt_dict_loc[sn][nt] = curr_modif_rates

            quantile_90 = np.quantile(curr_modif_rates, 0.9)
            quantile_95 = np.quantile(curr_modif_rates, 0.95)
            quantile_99 = np.quantile(curr_modif_rates, 0.99)

            row = i
            col = k

            if sn.startswith('c'):
                color = 'green'
            else:
                color = 'red'

            axs[row, col].hist(curr_modif_rates, bins=50, color=color, range=(0, xlim))
            if quantile_90 < xlim:
                axs[row, col].vlines(quantile_90, 0, ylim, color='#525252')
            if quantile_95 < xlim:
                axs[row, col].vlines(quantile_95, 0, ylim, color='#252525')
            if quantile_99 < xlim:
                axs[row, col].vlines(quantile_99, 0, ylim, color='#000000')
            axs[row, col].set_title("Sample: %s, nucleotide: %s" % (sn, nt), fontsize=16)
            axs[row, col].set_ylim(0, ylim)
            # axs[row, col].set_xlabel("number of positions", fontsize=18)
    fig.tight_layout()
    plt.show()
    return modif_rates_per_nt_dict_loc


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


def try_several_thresholds(inp_array_unf,
                           ref_vec_raw,
                           cov_vec_loc,
                           nt_to_num_loc,
                           sample_names_loc,
                           sample_pairs,
                           thresholds_list,
                           potential_thresholds,
                           min_coverage=20):
    thresholds_list = list(thresholds_list)
    cov_mask_loc = np.median(cov_vec_loc, axis=1) >= min_coverage
    inp_array = inp_array_unf.copy()
    inp_array = inp_array[:, cov_mask_loc, :]
    ref_vec_loc = ref_vec_raw.copy()
    ref_vec_loc = ref_vec_loc[cov_mask_loc, :]

    c_to_dms_ratios_array = np.zeros((len(thresholds_list), len(sample_pairs)))
    AC_median_ratios_array = np.zeros((len(thresholds_list), inp_array.shape[0]))
    median_modified_count = np.zeros((len(thresholds_list), inp_array.shape[0]))
    for i, thr in enumerate(thresholds_list):
        ratios_list, AC_meidan_ratios_ind, meidan_N_modified = apply_threshold(
            inp_array,
            ref_vec_loc,
            nt_to_num_loc,
            sample_pairs,
            thr)
        AC_median_ratios_array[i, :] = AC_meidan_ratios_ind
        c_to_dms_ratios_array[i, :] = np.array(ratios_list)
        median_modified_count[i, :] = meidan_N_modified

    fig, axs = plt.subplots(nrows=4, ncols=8)
    fig.set_size_inches(20, 17)

    median_modified_count_quantile_99 = np.quantile(median_modified_count, 0.99)

    for i in range(inp_array.shape[0]):
        curr_AC_fractions_list = AC_median_ratios_array[:, i]
        curr_median_modified = median_modified_count[:, i]
        curr_sample_name = sample_names_loc[i]

        if curr_sample_name.startswith('c'):
            color = 'green'
        else:
            color = 'red'

        axs[0, i].plot(thresholds_list, curr_AC_fractions_list, color=color)
        axs[0, i].set_title("%s AC/AVGT \n fraction" % (curr_sample_name))
        for t in potential_thresholds:
            axs[0, i].vlines(t, 0.5, 1, linestyles='dotted', linewidth=0.4)
        axs[0, i].set_ylim(0.5, 1)

        axs[1, i].plot(thresholds_list, curr_median_modified, color=color)
        axs[1, i].set_title("%s median N \n modified" % (curr_sample_name))
        for t in potential_thresholds:
            axs[1, i].vlines(t, 0,
                             median_modified_count_quantile_99,
                             linestyles='dotted', linewidth=0.4)
        axs[1, i].set_ylim(0, median_modified_count_quantile_99)

        if i < len(sample_pairs):
            curr_pair = sample_pairs[i]
            curr_pair_name = curr_pair[0]
            curr_dms = curr_pair[1]
            curr_c_to_dms_ratios = c_to_dms_ratios_array[:, i]

            axs[2, i].plot(thresholds_list, curr_c_to_dms_ratios, color='blue')
            axs[2, i].set_title("%s ratio c/d \n modified" % (curr_pair_name))
            for t in potential_thresholds:
                axs[2, i].vlines(t, 0, 1, linestyles='dotted', linewidth=0.4)
            axs[2, i].set_ylim(0, 1)

            expected_false_nts = curr_c_to_dms_ratios * median_modified_count[:, curr_dms]

            axs[3, i].plot(thresholds_list, expected_false_nts, color='blue')
            axs[3, i].set_title("%s expected n \n false positives" % (curr_sample_name))
            for t in potential_thresholds:
                axs[3, i].vlines(t, 0, 10, linestyles='dotted', linewidth=0.4)
            axs[3, i].set_ylim(0, 10)

    fig.tight_layout()
    plt.plot()

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
    inp_array[:, T_id] = 0
    inp_array[:, G_id] = 0
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


def plot_median_count_vs_correlation(mut_counts_array_loc,
                                     ref_vec_loc,
                                     sample_names_loc,
                                     sample_pairs_list,
                                     nt_to_num_loc,
                                     do_log_median=False,
                                     nts_of_interest=['A', 'C'],
                                     stdev=0.3):
    nts_list = sorted(list(nt_to_num_loc.keys()))
    fig, axs = plt.subplots(nrows=len(sample_pairs_list), ncols=len(nts_of_interest))
    fig.set_size_inches(24, 32)

    for k, samples_tuple in enumerate(sample_pairs_list):
        sample_1, sample_2 = samples_tuple
        sample_1_id = sample_names_loc.index(sample_1)
        sample_2_id = sample_names_loc.index(sample_2)

        for nt in nts_of_interest:
            nt_id = nt_to_num_loc[nt]
            curr_nt_mask = ref_vec_loc != nt_id
            sample_1_nt_array = mut_counts_array_loc[sample_1_id, :, :].copy()
            sample_2_nt_array = mut_counts_array_loc[sample_2_id, :, :].copy()
            sample_1_nt_array_masked = np.ma.array(sample_1_nt_array, mask=curr_nt_mask)
            sample_2_nt_array_masked = np.ma.array(sample_2_nt_array, mask=curr_nt_mask)
            median_mut_count_1 = np.ma.median(sample_1_nt_array_masked, axis=1)
            median_mut_count_2 = np.ma.median(sample_1_nt_array_masked, axis=1)
            min_medians = np.minimum(median_mut_count_1, median_mut_count_2)
            if do_log_median:
                min_medians = np.log(min_medians + 1)
            # to jitter dots like here: https://stackoverflow.com/questions/8671808/matplotlib-avoiding-overlapping-datapoints-in-a-scatter-dot-beeswarm-plot
            min_medians += np.random.randn(min_medians.shape[0]) * stdev
            quantile_99 = np.quantile(min_medians, 0.95)

            correlations_array = np.zeros_like(min_medians)
            for y in range(sample_1_nt_array.shape[0]):
                sample_1_counts = sample_1_nt_array[y]
                sample_2_counts = sample_2_nt_array[y]
                if np.std(sample_1_counts) == 0 or np.std(sample_2_counts) == 0:
                    correlations_array[y] = 0
                    continue  # if all values in one of the vectors are zero
                current_correlation, pv = scipy.stats.pearsonr(sample_1_counts, sample_2_counts)
                correlations_array[y] = current_correlation

            axs[k, nt_id].scatter(min_medians, correlations_array, s=5, alpha=0.2)
            axs[k, nt_id].set_title("%s vs %s ; %s" % (sample_1, sample_2, nt), fontsize=16)
            axs[k, nt_id].set_xlim(0, quantile_99)
            axs[k, nt_id].set_xlabel("Median number of mutations", fontsize=18)
            axs[k, nt_id].set_ylabel("Correlations", fontsize=18)

    fig.tight_layout()


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


def visualize_multiple_elements(inp_array,
                                ref_vec_loc,
                                nt_to_num_loc,
                                nucleotides=['A', 'T', 'G', 'C'],
                                threshold=np.nan):
    inp_array_copy = inp_array.copy()
    n_elements = inp_array_copy.shape[1]

    fig, axs = plt.subplots(nrows=n_elements, ncols=1)
    fig.set_size_inches(16, 2.5 * n_elements)
    for i in range(n_elements):
        curr_array = inp_array_copy[:, i, :]
        curr_ref_vec_loc = ref_vec_loc[i, :]
        curr_vmax = curr_array.max()

        nt_mask = np.zeros_like(curr_array).astype(np.bool)
        for nt in nucleotides:
            nt_mask = np.logical_or(nt_mask, curr_ref_vec_loc == nt_to_num_loc[nt])

        if not np.isnan(threshold):
            curr_array[curr_array < threshold] = 0

        curr_array[np.invert(nt_mask)] = 0
        axs[i].pcolor(curr_array, cmap='binary', vmax=curr_vmax)
        axs[i].invert_yaxis()
    plt.show()


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


def plot_read_umi_counts_before_after(read_before_dict, umis_before_dict,
                                      reads_after_dict, umis_after_dict,
                                      full_fragments_dict):
    assert (len(read_before_dict) == len(reads_after_dict))
    assert (len(umis_before_dict) == len(umis_after_dict))
    for sn in sorted(list(read_before_dict.keys())):
        cur_reads_before_dict = read_before_dict[sn]
        cur_reads_after_dict = reads_after_dict[sn]
        cur_umis_before_dict = umis_before_dict[sn]
        cur_umis_after_dict = umis_after_dict[sn]

        for el in full_fragments_dict:
            if el not in cur_reads_before_dict:
                cur_reads_before_dict[el] = 0
            if el not in cur_reads_after_dict:
                cur_reads_after_dict[el] = 0
            if el not in cur_umis_before_dict:
                cur_umis_before_dict[el] = {}
            if el not in cur_umis_after_dict:
                cur_umis_after_dict[el] = {}

        sorted_elements = sorted(list(full_fragments_dict.keys()))
        n_element = len(sorted_elements)
        read_counts_before_np = np.zeros(n_element)
        read_counts_after_np = np.zeros(n_element)
        umi_counts_before_np = np.zeros(n_element)
        umi_counts_after_np = np.zeros(n_element)

        for i, el in enumerate(sorted_elements):
            read_counts_before_np[i] = cur_reads_before_dict[el]
            read_counts_after_np[i] = cur_reads_after_dict[el]
            umi_counts_before_np[i] = len(cur_umis_before_dict[el])
            umi_counts_after_np[i] = len(cur_umis_after_dict[el])

        reads_max = max(read_counts_before_np.max(), read_counts_after_np.max())
        umis_max = max(umi_counts_before_np.max(), umi_counts_after_np.max())

        read_counts_after_np_nz = read_counts_after_np[read_counts_before_np != 0]
        read_counts_before_np_nz = read_counts_before_np[read_counts_before_np != 0]
        read_ratios = np.divide(read_counts_after_np_nz, read_counts_before_np_nz)
        median_read_ratio = np.median(read_ratios)

        umi_counts_after_np_nz = umi_counts_after_np[umi_counts_before_np != 0]
        umi_counts_before_np_nz = umi_counts_before_np[umi_counts_before_np != 0]
        umi_ratios = np.divide(umi_counts_after_np_nz, umi_counts_before_np_nz)
        median_umi_ratio = np.median(umi_ratios)

        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(20, 8)
        axs[0].scatter(read_counts_after_np, read_counts_before_np, s=0.3, alpha=0.3)
        axs[0].plot([0, reads_max], [0, reads_max], linewidth=0.5, color='black')
        axs[0].set_title("Number of reads per transcript before and after dedup \n sample %s" % sn, fontsize=24)
        axs[0].set_xlim(0, reads_max)
        axs[0].set_ylim(0, reads_max)
        axs[0].set_xlabel("After", fontsize=18)
        axs[0].set_ylabel("Before", fontsize=18)

        axs[1].scatter(umi_counts_after_np, umi_counts_before_np, s=0.3, alpha=0.3)
        axs[1].plot([0, umis_max], [0, umis_max], linewidth=0.5, color='black')
        axs[1].set_title("Number of UMIs per transcript before and after dedup \n sample %s" % sn, fontsize=24)
        axs[1].set_xlim(0, umis_max)
        axs[1].set_ylim(0, umis_max)
        axs[1].set_xlabel("After", fontsize=18)
        axs[1].set_ylabel("Before", fontsize=18)

        print("Sample %s: median read count ratio is %.2f, median UMI count ratio is %.1f" %
              (sn, median_read_ratio, median_umi_ratio))

        fig.tight_layout()
        plt.show()


def calculate_mediat_modifications_per_transcript(mut_counts_array_loc,
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