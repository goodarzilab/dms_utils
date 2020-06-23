import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dms_utils.utils.utils as utils
import networkx as nx


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


def print_subs_dels(ref_vec_loc, count_dict_loc,
                    dels_not_common,
                    tr_mean_coverage_loc,
                    num_to_nt_loc,
                    buffer=10,
                    min_coverage=20,
                    do_print_consensus=True):
    sample_names = sorted(list(count_dict_loc.keys()))
    d_x, d_y = np.where(dels_not_common == 1)
    segments_loc = utils.get_intervals_list_from_pairs(d_x, d_y)

    for segment in segments_loc:
        el_id = segment[0]
        start = segment[1]
        end = segment[2]
        start_ext = max(0, start - buffer)
        end_ext = min(ref_vec_loc.shape[1], end + buffer)
        print("Element number %d; start %d, end %d" % (el_id, start, end))
        reference_sequence_before = utils.interval_to_seq(ref_vec_loc[el_id][start_ext: start], num_to_nt_loc)
        reference_sequence = utils.interval_to_seq(ref_vec_loc[el_id][start: end], num_to_nt_loc)
        reference_sequence_after = utils.interval_to_seq(ref_vec_loc[el_id][end: end_ext], num_to_nt_loc)
        reference_sequence_string = "reference: %s - %s - %s" % (reference_sequence_before,
                                                                 reference_sequence,
                                                                 reference_sequence_after)
        print(reference_sequence_string)
        for i, sample in enumerate(sample_names):
            current_sequence_before = utils.interval_to_seq_from_data(count_dict_loc[sample][el_id, start_ext: start, :],
                                                                num_to_nt_loc)
            current_sequence = utils.interval_to_seq_from_data(count_dict_loc[sample][el_id, start: end, :], num_to_nt_loc)
            current_sequence_after = utils.interval_to_seq_from_data(count_dict_loc[sample][el_id, end: end_ext, :],
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
    reference_sequence = utils.interval_to_seq(ref_vec_loc[el_id], num_to_nt_loc)
    reference_sequence_string = "reference: %s" % reference_sequence
    print(reference_sequence_string)
    for i, sample in enumerate(sample_names):
        current_sequence = utils.interval_to_seq_from_data(count_dict_loc[sample][el_id, :, :], num_to_nt_loc)
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


def plot_median_count_vs_correlation(mut_counts_array_loc,
                                     ref_vec_loc,
                                     sample_names_loc,
                                     sample_pairs_list,
                                     nt_to_num_loc,
                                     do_log_median=False,
                                     nts_of_interest=['A', 'C'],
                                     stdev=0.3):
    # nts_list = sorted(list(nt_to_num_loc.keys()))
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


def plot_read_umi_counts_before_after(read_before_dict, umis_before_dict,
                                      reads_after_dict, umis_after_dict,
                                      full_fragments_dict,
                                      do_quantiles = False,
                                      quantile = 0.99):
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
        if do_quantiles:
            reads_max = max(np.quantile(read_counts_before_np, quantile), np.quantile(read_counts_after_np, quantile))
            umis_max = max(np.quantile(umi_counts_before_np, quantile), np.quantile(umi_counts_after_np, quantile))

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
        ratios_list, AC_meidan_ratios_ind, meidan_N_modified = utils.apply_threshold(
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


def plot_graph_weighted_enges(G, ax,
                              attribute_name = 'weight',
                              cmap = plt.cm.Blues):
    if cmap is None:
        edge_color = None
    else:
        edge_color = nx.get_edge_attributes(G,attribute_name).values()

    nx.draw_spring(G,
       edge_color=edge_color,
       node_size=1,
       edge_cmap=cmap,
       with_labels=True, ax = ax)
    return ax


def plot_graph_colored_nodes(G, ax,
                              attribute_name,
                              cmap=plt.cm.Set1,
                              node_size = 80):
    if cmap is None:
        node_color = None
    else:
        node_color = list(nx.get_node_attributes(G, attribute_name).values())

    nx.draw_spring(G,
                   node_color=node_color,
                   node_size=node_size,
                   cmap=cmap,
                   with_labels=True, ax=ax)
    return ax


def plot_network_clusters_modified(graph, partition, position, ax,
                                   node_size=200):
    color_array = ['r', 'b', 'g', 'c', 'm', 'y', 'k',
             '0.8', '0.2', '0.6', '0.4', '0.7', '0.3', '0.9', '0.1', '0.5']
    partition = partition.communities
    n_communities = min(len(partition), len(color_array))
    if position is None:
        position = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, position, ax = ax, node_size=node_size, node_color='w')
    nx.draw_networkx_edges(graph, position, ax = ax, alpha=.5)
    for i in range(n_communities):
        if len(partition[i]) > 0:
            size = (n_communities - i) * node_size
            nx.draw_networkx_nodes(graph, position, ax = ax, node_size=size,
                                         nodelist=partition[i], node_color=color_array[i])
    nx.draw_networkx_labels(graph, position, ax = ax, labels={node: str(node) for node in graph.nodes()})


def plot_network_clusters_ground_truth(graph, position, ax,
                                   attribute_name,
                                   node_size=200,
                                   cmap=plt.cm.Set1,
                                   edge_cmap=None):
    if cmap is None:
        node_color = None
    else:
        node_color = list(nx.get_node_attributes(graph, attribute_name).values())

    if edge_cmap is None:
        edge_color = None
    else:
        edge_color = nx.get_edge_attributes(graph,'weight').values()

    if position is None:
        position = nx.spring_layout(graph)
    nx.draw_networkx_edges(graph, position, ax = ax, alpha=.5, edge_color=edge_color, edge_cmap=edge_cmap)
    nx.draw_networkx_nodes(graph, position, ax = ax, node_size=node_size, node_color=node_color)
    nx.draw_networkx_labels(graph, position, ax = ax, labels={node: str(node) for node in graph.nodes()})
