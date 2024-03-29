import os
import numpy as np
import pandas as pd
import json

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


def read_bitvector_to_df(inp_file):
    bit_df_loc = pd.read_csv(inp_file, sep='\t',
                             index_col = 0, skiprows = 2)
    return bit_df_loc


def read_bitvector_reference_sequence(inp_file):
    with open(inp_file, 'r') as rf:
        lines_array = rf.readlines()
    ref_seq_loc = lines_array[0].split('\t')[-1].replace('\n','')
    return ref_seq_loc


def parse_draco_file(inp_filename,
                     id_of_interest):
    handle = open(inp_filename, )
    json_data = json.load(handle)
    ids_matching = []
    for i, tr in enumerate(json_data['transcripts']):
        if tr['id'] == id_of_interest:
            ids_matching.append(i)
    assert len(ids_matching) == 1
    id = ids_matching[0]
    windows_list = json_data['transcripts'][id]['windows']
    full_sequence = json_data['transcripts'][id]['sequence']
    return windows_list, full_sequence

