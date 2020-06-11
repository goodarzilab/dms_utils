import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dms_utils.utils.utils as utils
import dms_utils.utils.dreem_original_functions as dof
import time



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
                     masked_postions = [],
                     start = np.nan, end = np.nan,
                     paired = False,
                     NUM_RUNS=2,
                     MIN_ITS=10,
                     CONV_CUTOFF=0.5,
                     CPUS=2
                     ):
    refs_seq = dof.Parse_FastaFile(ref_file)
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
                    sur_bases, del_bit, miss_info, bases,
                    refs_seq[ref_name], masked_postions)

    close_open_files(files)

    INFO_THRESH, SIG_THRESH, inc_TG, NORM_PERC_BASES, \
    NUM_RUNS, MIN_ITS, MAX_K, CONV_CUTOFF, CPUS, struct = define_global_variables_within_EM_clustering(
                                                                                                NUM_RUNS=2,
                                                                                                MIN_ITS=10,
                                                                                                CONV_CUTOFF = 0.5,
                                                                                                CPUS = 2
                                                                                                )
    launch_EM_clustering(sample_name, ref_name, start, end, out_folder, ref_file,
                         INFO_THRESH, SIG_THRESH, inc_TG, NORM_PERC_BASES,
                         NUM_RUNS, MIN_ITS, MAX_K, CONV_CUTOFF, CPUS, struct)


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
                    sur_bases, del_bit, miss_info, bases,
                    ref_seq, masked_postions):
    dof.Process_SamFile(sam_file, paired, refs_seq, start, end,
                    cov_bases, info_bases, mod_bases,
                    mut_bases, delmut_bases, num_reads, files,
                    ref_name, phred_qscore,
                    qscore_cutoff, nomut_bit, ambig_info,
                    sur_bases, del_bit, miss_info, bases,
                    ref_seq, masked_postions)


def define_global_variables_within_EM_clustering(INFO_THRESH = 0.05,
                                                 SIG_THRESH = 0.005, # see the main paper text
                                                 inc_TG = False,
                                                 # The DMS signal is normalized such that the median of the top ten most-reactive positions is set to 1.0
                                                NORM_PERC_BASES = 10,
                                                NUM_RUNS = 2,
                                                MIN_ITS = 10, # change it depending on how many iterations do you want to do
                                                MAX_K = 2, # Max clusters to work on
                                                CONV_CUTOFF = 0.5,
                                                 CPUS = 2,
                                                 struct = True
                                                 ):
    return INFO_THRESH, SIG_THRESH, inc_TG, NORM_PERC_BASES, \
           NUM_RUNS, MIN_ITS, MAX_K, CONV_CUTOFF, CPUS, struct



def launch_EM_clustering(sample_name, name, start, end, out_folder, ref_file,
                         INFO_THRESH, SIG_THRESH, inc_TG, NORM_PERC_BASES,
                         NUM_RUNS, MIN_ITS, MAX_K, CONV_CUTOFF, CPUS, struct):
        bvfile_basename = '{}_{}_{}_{}'.format(sample_name, name, start, end)
        outplot_dir = os.path.join(out_folder, bvfile_basename) + '/'
        if not os.path.exists(outplot_dir):
            os.makedirs(outplot_dir)
        else:  # Folder exists
            if os.path.exists(outplot_dir + 'log.txt'):  # Log file exists
                print('EM Clustering already done for', bvfile_basename)
                return

        wind_size = int(end) - int(start)
        norm_bases = int((wind_size * NORM_PERC_BASES) / 100)

        # Read the bit vector file and do the filtering
        input_file = os.path.join(out_folder, bvfile_basename + '_bitvectors.txt')

        X = dof.Load_BitVectors(input_file, INFO_THRESH, SIG_THRESH,
                                     inc_TG, out_folder)

        K = 1  # Number of clusters
        cur_BIC = float('inf')  # Initialize BIC
        BIC_failed = False  # While test is not passed
        while not BIC_failed and K <= MAX_K:
            print('Working on K =', K)

            RUNS = NUM_RUNS if K != 1 else 1  # Only 1 Run for K=1
            ITS = MIN_ITS if K != 1 else 10  # Only 10 iters for K=1

            for run in range(1, RUNS + 1):
                print('Run number:', run)
                dof.Run_EMJob(X, bvfile_basename, ITS,
                                     CONV_CUTOFF,
                                     outplot_dir, K, CPUS, run)

            # Processing of results from the EM runs
            input_dir = ''
            dof.Post_Process(bvfile_basename, ref_file, K, RUNS,
                                        cur_BIC, norm_bases, struct,
                                        input_dir, outplot_dir)

            # Check BIC
            latest_BIC = dof.Collect_BestBIC(bvfile_basename, K,
                                                        outplot_dir)
            if latest_BIC > cur_BIC:  # BIC test has failed
                BIC_failed = True
            cur_BIC = latest_BIC  # Update BIC

            K += 1  # Move on to next K

        # Write params to log file
        time_taken = 0
        dof.Log_File(bvfile_basename, NUM_RUNS, MIN_ITS,
                          CONV_CUTOFF, INFO_THRESH, SIG_THRESH, inc_TG,
                          norm_bases, K - 2, time_taken, outplot_dir)