import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dms_utils.utils.utils as utils
import dms_utils.utils.dreem_original_functions as dof


def Parse_FastaFile(fasta_file):
    from Bio import SeqIO
    refs_seq = {}
    with open(fasta_file, 'rU') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            refs_seq[record.id] = str(record.seq)
    return refs_seq


# def define_hardcoded_dreem_parameters():
#     # Inputs for Step 2 - Bit Vector creation
#     picard_path = CUR_DIR + '/picard.jar'  # Picard jar file in cur dir
#     qscore_file = CUR_DIR + '/phred_ascii.txt'  # ASCII char - Q score map
#     SUR_BASES = 10  # Bases surrounding a deletion on each side
#     QSCORE_CUTOFF = 20  # Qscore cutoff for a valid base
#
#     # Inputs for Step 3 - EM Clustering
#     INFO_THRESH = 0.05  # Threshold for informative bits
#     CONV_CUTOFF = 0.5  # Diff in log like for convergence
#     NUM_RUNS = 10  # Number of independent EM runs per K
#     struct = True  # Run structure prediction or not


def launch_stemAC():
    sample_name = "downsampled_1"
    ref_file = "/rumi/shams/khorms/projects/SNIP_switchers/published_DMSseq_data/tomezsko_2020/stemsAC/StemA_C/downsampling/06_04_20/181012RRou_D18-9764_downsampled_1.bam"
    ref_name = "stemAC"
    define_global_variables_within_bit_vector(sample_name, ref_file, ref_name)




def define_global_variables_within_bit_vector(seq_of_interest,
                                              start = np.nan, end = np.nan,
                                              paired = False,
                                              miss_info = '.',
                                              ambig_info= '?',
                                              nomut_bit = '0',
                                              del_bit= '1',
                                              bases = ['A', 'T', 'G', 'C'],
                                              sur_bases = 10,
                                              qscore_cutoff = 20,
                                              ):
    start = 1
    end = len(seq_of_interest) + 1
    #phred_qscore = BitVector_Functions.Parse_PhredFile(qscore_file)
    return start, end, paired, miss_info, ambig_info, \
                nomut_bit, del_bit, bases, \
                sur_bases, qscore_cutoff

def launch_something(sample_name, ref_file, ref_name,
                     sam_file, out_folder):
    refs_seq = Parse_FastaFile(ref_file)
    seq_of_interest = refs_seq[ref_name]
    start, end, paired, miss_info, ambig_info, \
    nomut_bit, del_bit, bases, \
    sur_bases, qscore_cutoff = define_global_variables_within_bit_vector(seq_of_interest)

    mod_bases, mut_bases, delmut_bases, info_bases, cov_bases, files, num_reads = initialize_plotting_variables(
                                  ref_name, sample_name, seq_of_interest,
                                  out_folder,
                                  start, end, bases)
    compute_bit_vectors(
                    sam_file, paired, refs_seq, start, end,
                    cov_bases, info_bases, mod_bases,
                    mut_bases, delmut_bases, num_reads, files)



#def write_ref_files():
def initialize_plotting_variables(name, sample_name, seq_of_interest,
                                  out_folder,
                                  start, end, bases):
    # Initialize plotting variables
    mod_bases, mut_bases, delmut_bases = {}, {}, {}
    info_bases, cov_bases = {}, {}
    files, num_reads = {}, {}
    num_reads[name] = 0
    mod_bases[name], mut_bases[name], delmut_bases[name] = {}, {}, {}
    info_bases[name], cov_bases[name] = {}, {}
    for base in bases:
        mod_bases[name][base] = {}
        for pos in range(start, end + 1):
            mod_bases[name][base][pos] = 0
    for pos in range(start, end + 1):
        mut_bases[name][pos], delmut_bases[name][pos] = 0, 0
        info_bases[name][pos], cov_bases[name][pos] = 0, 0
        # Write header lines to output text file
        file_base_name = sample_name + '_' + name + '_' + str(start) + \
                         '_' + str(end)
        output_txt_filename = os.path.join(out_folder, file_base_name + '_bitvectors.txt')
        files[name] = open(output_txt_filename, 'w')
        files[name].write('@name' + '\t' + name + ';' + name + '\t' +
                          seq_of_interest[start - 1:end] + '\n')
        files[name].write('@coordinates:length' + '\t' + str(start) + ',' +
                          str(end) + ':' + str(end - start + 1) + '\n')
        files[name].write('Query_name\tBit_vector\tN_Mutations\n')

    return mod_bases, mut_bases, delmut_bases, info_bases, cov_bases, files, num_reads


def compute_bit_vectors(sam_file, paired, refs_seq, start, end,
                    cov_bases, info_bases, mod_bases,
                    mut_bases, delmut_bases, num_reads, files):
    dof.Process_SamFile(sam_file, paired, refs_seq, start, end,
                    cov_bases, info_bases, mod_bases,
                    mut_bases, delmut_bases, num_reads, files)

