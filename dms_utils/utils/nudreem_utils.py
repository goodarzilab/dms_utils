from .glob_vars import bitstring_letters_to_num
import pandas as pd
import numpy as np


def convert_bitstring_to_numpy(inp_string):
    out_array = np.zeros(len(inp_string), dtype=np.int8)
    for i in range(len(inp_string)):
        out_array[i] = bitstring_letters_to_num[inp_string[i]]
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



def combine_several_masks():
    pass