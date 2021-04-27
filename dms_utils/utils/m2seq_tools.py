import numpy as np
import os
import pandas as pd

def generate_standard_commands_per_sample(
                                folder_of_interest,
                                output_subfolder,
                                sample,
                                elim_run_id,
                                file_names_array,
                                sequence,
                                structure,
                                offset,
                                primer_length,
                                ref_segment,
                                data_types_list,
                                sample_labels,
                                align_string = "1:2, 3:4, 5, 6, 7, 8"):
    commands = []
    # move to the folder of interest
    move_to_folder_command = "cd %s" % (folder_of_interest)
    commands.append(move_to_folder_command)
    # step 1: quick_look
    file_numbers = ", ".join(file_names_array)
    quick_look_command = "d_align_%s = quick_look({'%s'},[],[%s]);" % (sample, elim_run_id, file_numbers)
    commands.append(quick_look_command)
    # step 2: refine the alignment
    align_blocks_command = "align_blocks_%s = {%s};" % (sample, align_string)
    fine_align_command = "d_align_%s_dp_fine = align_by_DP_fine(d_align_%s, align_blocks_%s);" % (sample, sample, sample)
    commands.append(align_blocks_command)
    commands.append(fine_align_command)
    # step 3: input sequence and structure
    seq_define_command = "sequence_%s = '%s';" %  (sample, sequence)
    seq_replace_command = "sequence_%s = strrep(sequence_%s, 'T', 'U');" % (sample, sample)
    commands.append(seq_define_command)
    commands.append(seq_replace_command)
    structure_define_command = "structure_%s = '%s';" % (sample, structure)
    data_types_list_standardised = [x.replace("_dilution", "") for x in data_types_list]
    data_types_define_command = "data_types_%s = [%s];" % (sample, ", ".join(["{'%s'}" % x for x in data_types_list_standardised]))
    offset_command = "offset_%s = -%d;" % (sample, offset)
    primer_length_command = "primer_length_%s = %d;" % (sample, primer_length)
    first_nucl_command = "first_RT_nucleotide_%s = length(sequence_%s) - primer_length_%s + offset_%s;" \
                         % (sample, sample, sample, sample)
    commands.append(structure_define_command)
    commands.append(data_types_define_command)
    commands.append(offset_command)
    commands.append(primer_length_command)
    commands.append(first_nucl_command)
    # step 4: initialize xsel
    xsel_initialize_command = "xsel_%s = []; clf;" % (sample)
    xsel_calculate_command = "[xsel_%s, seqpos_%s, area_pred_%s] = annotate_sequence(d_align_%s_dp_fine, " \
                             "xsel_%s, sequence_%s, offset_%s, data_types_%s, first_RT_nucleotide_%s, structure_%s);" \
                             % (sample, sample, sample, sample, sample,
                                sample, sample, sample, sample, sample)
    commands.append(xsel_initialize_command)
    commands.append(xsel_calculate_command)
    # step 5: fit the peaks
    peak_fitting_command = "[area_peak_%s, darea_peak_%s] = fit_to_gaussians(d_align_%s, xsel_%s);" \
                           % (sample, sample, sample, sample)
    commands.append(peak_fitting_command)
    # step 6: data correction
    saturated_indices = [str(i + 1) for i, x in enumerate(data_types_list) if x == "1M7"]
    diluted_indices = [str(i + 1) for i, x in enumerate(data_types_list) if x == "1M7_dilution"]
    nomod_saturated_indices = [str(i + 1) for i, x in enumerate(data_types_list) if x == "nomod"]
    nomod_diluted_indices = [str(i + 1) for i, x in enumerate(data_types_list) if x == "nomod_dilution"]

    saturated_numpy_indices = np.logical_or(data_types_list == '1M7', data_types_list == 'nomod')
    diluted_numpy_indices = np.logical_or(data_types_list == '1M7_diluted', data_types_list == 'nomod_diluted')
    either_indices = np.logical_or(saturated_numpy_indices, diluted_numpy_indices)
    sample_flat_subset = sample_labels[either_indices]
    classnames, indices = np.unique(sample_flat_subset, return_inverse = True)
    bkg_col_array = [str(x+1) for x in indices]
    data_types_anno = ["'%s'" % x.replace("_diluted", "") for x in data_types_list[saturated_numpy_indices]]
    # nomod_saturated_array_command = "saturated_array_nomod_%s = [mean(area_peak_%s(:, [%s]), 2)];" % (sample, sample, ", ".join(nomod_saturated_indices))
    # nomod_diluted_array_command = "diluted_array_nomod_%s = [mean(area_peak_%s(:, [%s]), 2)];" % (sample, sample, ", ".join(nomod_diluted_indices))
    saturated_array_command = "saturated_array_%s = area_peak_%s(:, [%s]);" % (sample, sample, ", ".join(nomod_saturated_indices + saturated_indices))
    diluted_array_command = "diluted_array_%s = area_peak_%s(:, [%s]);" % (sample, sample, ", ".join(nomod_diluted_indices + diluted_indices))
    saturated_error_command = "saturated_error_%s = darea_peak_%s(:, [%s]);" % (sample, sample, ", ".join(nomod_saturated_indices + saturated_indices))
    diluted_error_command = "diluted_error_%s = darea_peak_%s(:, [%s]);" % (sample, sample, ", ".join(nomod_diluted_indices + diluted_indices))
    bkg_col_command = "bkg_col_%s = [%s];" % (sample, ", ".join(bkg_col_array))
    ref_segment_command = "ref_segment_%s = '%s';" % (sample, ref_segment)
    ref_peak_command = "ref_peak_%s = get_ref_peak(sequence_%s, ref_segment_%s, offset_%s);" \
                       % (sample, sample, sample, sample)
    sd_cutoff_command = "sd_cutoff = 1.5;"
    norm_command = "[normalized_reactivity_%s, normalized_error_%s, seqpos_out_%s] = get_reactivities(" \
                       "saturated_array_%s, diluted_array_%s, saturated_error_%s, diluted_error_%s, bkg_col_%s," \
                       " ref_peak_%s, seqpos_%s, [], {%s}, sequence_%s, " \
                                                            "offset_%s, sd_cutoff);" % (sample, sample, sample,
                                                                                           sample, sample, sample,
                                                                                           sample, sample, sample,
                                                                                           sample,
                                                                                           ", ".join(data_types_anno),
                                                                                           sample, sample)
    # commands.append(nomod_saturated_array_command)
    # commands.append(nomod_diluted_array_command)
    commands.append(ref_segment_command)
    commands.append(ref_peak_command)
    commands.append(sd_cutoff_command)
    commands.append(saturated_array_command)
    commands.append(diluted_array_command)
    commands.append(saturated_error_command)
    commands.append(diluted_error_command)
    commands.append(bkg_col_command)
    commands.append(norm_command)
    # step 7: error estimation
    # for each sample individually
    for i, sn in enumerate(classnames):
        sub_index = np.where(indices == i)[0]
        sub_index = sub_index[1:]  # to remove the nomod
        sub_index_string = [str(x+1) for x in sub_index]
        error_est_command = "[d_1M7_minus_%s, da_1M7_minus_%s, flags_%s] = average_data_filter_outliers(" \
                        "normalized_reactivity_%s(:, [%s]), normalized_error_%s(:, [%s]), []," \
                        " seqpos_out_%s, sequence_%s, offset_%s); " % (sn, sn, sn,
                                                                       sample, ", ".join(sub_index_string),
                                                                       sample, ", ".join(sub_index_string),
                                                                       sample,sample, sample)
        commands.append(error_est_command)
    # step 8: write output files
    for sn in classnames:
        make_shape_filename_command = "curr_shape_file = fullfile('%s', '%s', '%s_reactivity.shape')" % \
                                      (folder_of_interest, output_subfolder, sn)
        save_shape_command = "csvwrite(curr_shape_file, d_1M7_minus_%s)" % sn
        make_error_filename_command = "curr_shape_file = fullfile('%s', '%s', '%s_error.shape')" % \
                                      (folder_of_interest, output_subfolder, sn)
        save_error_command = "csvwrite(curr_shape_file, da_1M7_minus_%s)" % sn
        commands.append(make_shape_filename_command)
        commands.append(save_shape_command)
        commands.append(make_error_filename_command)
        commands.append(save_error_command)

    commands_full_string = "\n".join(commands)
    print(commands_full_string)
    print()


def find_spans_condition_array(condition_flat):
    spans_list = []
    for element in np.unique(condition_flat):
        where_array = np.where(condition_flat == element)[0]
        where_min = where_array[0]
        where_max = where_array[-1]
        spans_list.append("%d:%d" % (where_min + 1, where_max + 1))
    string_spans = ", ".join(spans_list)
    return string_spans




def load_shape_vectors(folder,
                       sample_names):
    shape_dict = {}
    error_dict = {}
    for sn in sample_names:
        shape_filename = os.path.join(folder, '%s_reactivity.shape' % sn)
        error_filename = os.path.join(folder, '%s_error.shape' % sn)
        shape_array = pd.read_csv(shape_filename, header = None)[0].values
        error_array = pd.read_csv(error_filename, header = None)[0].values
        shape_dict[sn] = shape_array
        error_dict[sn] = error_array

    return shape_dict, error_dict


def shape_vectors_to_array(shape_dict, error_dict,
                       sample_names):
    shape_array = np.zeros((len(sample_names), shape_dict[sample_names[0]].shape[0]))
    errors_array = np.zeros((len(sample_names), shape_dict[sample_names[0]].shape[0]))
    for i, sn in enumerate(sample_names):
        shape_array[i, :] = shape_dict[sample_names[i]]
        errors_array[i, :] = error_dict[sample_names[i]]
    return shape_array, errors_array