

def generate_standard_commands_per_sample(
                                sample,
                                elim_run_id,
                                file_names_array,
                                sequence,
                                structure,
                                offset,
                                primer_length,
                                ref_segment,
                                data_types_list,
                                align_string = "1:2, 3:4, 5, 6, 7, 8"):
    commands = []
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
    data_types_define_command = "data_types_%s = [%s];" % (sample, ", ".join(["{'%s'}" % x for x in data_types_list]))
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
    saturated_indices = [str(i + 1) for i, x in enumerate(data_types_list) if x == "SHAPE"]
    diluted_indices = [str(i + 1) for i, x in enumerate(data_types_list) if x == "SHAPE_dilution"]
    nomod_saturated_indices = [str(i + 1) for i, x in enumerate(data_types_list) if x == "NM"]
    nomod_diluted_indices = [str(i + 1) for i, x in enumerate(data_types_list) if x == "NM_dilution"]
    bkg_col_array = ["1"] * len(saturated_indices)
    saturated_array_command = "saturated_array_%s = area_peak_%s(:, [%s]);" % (sample, sample, ", ".join(saturated_indices))
    diluted_array_command = "diluted_array_%s = area_peak_%s(:, [%s]);" % (sample, sample, ", ".join(diluted_indices))
    saturated_error_command = "saturated_error_%s = darea_peak_%s(:, [%s]);" % (sample, sample, ", ".join(saturated_indices))
    diluted_error_command = "diluted_error_%s = darea_peak_%s(:, [%s]);" % (sample, sample, ", ".join(diluted_indices))
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
                                                                                       ", ".join(["'SHAPE'"] * len(saturated_indices)),
                                                                                       sample, sample)
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
    error_est_command = "[d_1M7_minus_%s, da_1M7_minus_%s, flags_%s] = average_data_filter_outliers(" \
                        "normalized_reactivity_%s, normalized_error_%s, []," \
                        " seqpos_out_%s, sequence_%s, offset_%s); " % (sample, sample, sample,
                                                                                       sample, sample, sample,
                                                                                       sample, sample)
    commands.append(error_est_command)
    commands_full_string = "\n".join(commands)
    print(commands_full_string)
    print()
