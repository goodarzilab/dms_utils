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
                                              miss_info = '.',
                                              ambig_info= '?',
                                              nomut_bit = '0',
                                              del_bit= '1',
                                              bases = ['A', 'T', 'G', 'C'],
                                              sur_bases = 10,
                                              qscore_cutoff = 20,
                                              ):
    if np.isnan(start):
        start = 1
    if np.isnan(end):
        end = len(seq_of_interest) + 1
    #phred_qscore = BitVector_Functions.Parse_PhredFile(qscore_file)
    return start, end, miss_info, ambig_info, \
                nomut_bit, del_bit, bases, \
                sur_bases, qscore_cutoff


def close_open_files(files):
    for name in files:
        files[name].close()


def launch_something(sample_name, ref_file, ref_name,
                     sam_file, out_folder, qscore_file,
                     start = np.nan, end = np.nan,
                     paired = False):
    refs_seq = Parse_FastaFile(ref_file)
    seq_of_interest = refs_seq[ref_name]
    phred_qscore = dof.Parse_PhredFile(qscore_file)
    start, end, miss_info, ambig_info, \
    nomut_bit, del_bit, bases, \
    sur_bases, qscore_cutoff = define_global_variables_within_bit_vector(seq_of_interest, start, end)

    mod_bases, mut_bases, delmut_bases, info_bases, cov_bases, files, num_reads = initialize_plotting_variables(
                                  ref_name, sample_name, seq_of_interest,
                                  out_folder,
                                  start, end, bases)
    compute_bit_vectors(
                    sam_file, paired, refs_seq, start, end,
                    cov_bases, info_bases, mod_bases,
                    mut_bases, delmut_bases, num_reads, files,
                    ref_name, phred_qscore,
                    qscore_cutoff, nomut_bit, ambig_info,
                    sur_bases, del_bit, miss_info, bases)

    close_open_files(files)





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
                    mut_bases, delmut_bases, num_reads, files,
                    ref_name, phred_qscore,
                    qscore_cutoff, nomut_bit, ambig_info,
                    sur_bases, del_bit, miss_info, bases):
    dof.Process_SamFile(sam_file, paired, refs_seq, start, end,
                    cov_bases, info_bases, mod_bases,
                    mut_bases, delmut_bases, num_reads, files,
                    ref_name, phred_qscore,
                    qscore_cutoff, nomut_bit, ambig_info,
                    sur_bases, del_bit, miss_info, bases)


#def define_global_variables_within_EM_clustering():

#
# def launch_EM_clustering():
#     for ref in refs_seq:  # Each seq in the ref genome
#
#         if ref != ref_name:
#             continue
#
#         start_time = time.time()
#
#         bvfile_basename = '{}_{}_{}_{}'.format(sample_name, ref, START, END)
#         outplot_dir = outfiles_dir + bvfile_basename + '/'
#         if not os.path.exists(outplot_dir):
#             os.makedirs(outplot_dir)
#         else:  # Folder exists
#             if os.path.exists(outplot_dir + 'log.txt'):  # Log file exists
#                 print('EM Clustering already done for', bvfile_basename)
#                 return
#
#         wind_size = int(END) - int(START)
#         norm_bases = int((wind_size * NORM_PERC_BASES) / 100)
#
#         # Read the bit vector file and do the filtering
#         input_file = output_dir + '/BitVector_Files/' + bvfile_basename + \
#             '_bitvectors.txt'
#         X = EM_Files.Load_BitVectors(input_file, INFO_THRESH, SIG_THRESH,
#                                      inc_TG, output_dir)
#
#         K = 1  # Number of clusters
#         cur_BIC = float('inf')  # Initialize BIC
#         BIC_failed = False  # While test is not passed
#         while not BIC_failed and K <= MAX_K:
#             print('Working on K =', K)
#
#             RUNS = NUM_RUNS if K != 1 else 1  # Only 1 Run for K=1
#             ITS = MIN_ITS if K != 1 else 10  # Only 10 iters for K=1
#
#             for run in range(1, RUNS + 1):
#                 print('Run number:', run)
#                 Run_EMJobs.Run_EMJob(X, bvfile_basename, ITS, INFO_THRESH,
#                                      CONV_CUTOFF, SIG_THRESH,
#                                      outplot_dir, K, CPUS, run)
#
#             # Processing of results from the EM runs
#             EM_CombineRuns.Post_Process(bvfile_basename, K, RUNS,
#                                         cur_BIC, norm_bases, struct,
#                                         input_dir, outplot_dir)
#
#             # Check BIC
#             latest_BIC = EM_CombineRuns.Collect_BestBIC(bvfile_basename, K,
#                                                         outplot_dir)
#             if latest_BIC > cur_BIC:  # BIC test has failed
#                 BIC_failed = True
#             cur_BIC = latest_BIC  # Update BIC
#
#             K += 1  # Move on to next K
#
#         end_time = time.time()
#         time_taken = round((end_time - start_time) / 60, 2)
#         print('Time taken:', time_taken, 'mins')
#
#         # Write params to log file
#         EM_Plots.Log_File(bvfile_basename, NUM_RUNS, MIN_ITS,
#                           CONV_CUTOFF, INFO_THRESH, SIG_THRESH, inc_TG,
#                           norm_bases, K - 2, time_taken, outplot_dir)