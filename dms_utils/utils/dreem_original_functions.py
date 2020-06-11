import numpy as np
import scipy.special
from scipy.optimize import newton_krylov
from multiprocessing.dummy import Pool as ThreadPool
import math
import pandas as pd
import os
import plotly
import plotly.graph_objs as go
from plotly import tools
from statistics import median as med
import datetime
from scipy.stats import linregress


def Parse_FastaFile(fasta_file):
    from Bio import SeqIO
    refs_seq = {}
    with open(fasta_file, 'rU') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            refs_seq[record.id] = str(record.seq)
    return refs_seq


def Create_FastaFile(filename, ref_name, seq):
    """
    Create a file in FASTA format that's used by BLAST
    Args:
        filename (string): Output file name
        ref_name (string): Name of ref
        seq (string): Sequence to write to file
    """
    outfile = open(filename, 'w')
    outfile.write('>' + ref_name + '\n')
    outfile.write(seq + '\n')
    outfile.close()

def GenerateBitVector_Single(mate, refs_seq, phred_qscore,
                             cov_bases, info_bases, mod_bases,
                             mut_bases, delmut_bases, num_reads, files,
                             start, end,
                            qscore_cutoff, nomut_bit, ambig_info,
                            sur_bases, del_bit, miss_info, bases,
                             ref_seq, masked_postions):
    """
    Create a bitvector for single end sequencing.
    """
    bit_vector = Convert_Read(mate, refs_seq, phred_qscore,
                 qscore_cutoff, nomut_bit, ambig_info,
                 sur_bases, del_bit, miss_info)
    Plotting_Variables(mate.QNAME, mate.RNAME, bit_vector, start, end,
                       cov_bases, info_bases, mod_bases, mut_bases,
                       delmut_bases, num_reads, files,
                       miss_info, ambig_info, bases, del_bit,
                       ref_seq, masked_postions)


def GenerateBitVector_Paired(mate1, mate2, start, end, refs_seq, phred_qscore,
                             cov_bases, info_bases, mod_bases, mut_bases,
                             delmut_bases, num_reads, files,
                             qscore_cutoff, nomut_bit, ambig_info,
                             sur_bases, del_bit, miss_info, bases,
                             ref_seq, masked_postions):
    """
    Create a bitvector for paired end sequencing.
    """
    bitvector_mate1 = Convert_Read(mate1, refs_seq, phred_qscore,
                                   qscore_cutoff, nomut_bit, ambig_info,
                                   sur_bases, del_bit, miss_info)
    bitvector_mate2 = Convert_Read(mate2, refs_seq, phred_qscore,
                                   qscore_cutoff, nomut_bit, ambig_info,
                                   sur_bases, del_bit, miss_info)
    bit_vector = Combine_Mates(bitvector_mate1, bitvector_mate2,
                               bases, nomut_bit, ambig_info)
    Plotting_Variables(mate1.QNAME, mate1.RNAME, bit_vector, start, end,
                       cov_bases, info_bases, mod_bases, mut_bases,
                       delmut_bases, num_reads, files,
                       miss_info, ambig_info, bases, del_bit,
                       ref_seq, masked_postions)


def Process_SamFile(sam_file, paired, refs_seq, start, end,
                    cov_bases, info_bases, mod_bases,
                    mut_bases, delmut_bases, num_reads, files,
                    ref_name, phred_qscore,
                    qscore_cutoff, nomut_bit, ambig_info,
                    sur_bases, del_bit, miss_info, bases,
                    ref_seq, masked_postions):
    """
    Read SAM file and generate bit vectors.
    """
    ignore_lines = len(refs_seq.keys()) + 2
    sam_fileobj = open(sam_file, 'r')
    for line_index in range(ignore_lines):  # Ignore header lines
        sam_fileobj.readline()
    while True:
        try:
            if paired:
                line1, line2 = next(sam_fileobj), next(sam_fileobj)
                line1, line2 = line1.strip().split(), line2.strip().split()
                mate1 = Mate(line1)
                mate2 = Mate(line2)
                assert mate1.PNEXT == mate2.POS and \
                    mate1.RNAME == mate2.RNAME and mate1.RNEXT == "="
                assert mate1.QNAME == mate2.QNAME and mate1.MAPQ == mate2.MAPQ
                if mate1.RNAME == ref_name:
                    GenerateBitVector_Paired(mate1, mate2, start, end, refs_seq,
                                             phred_qscore, cov_bases,
                                             info_bases, mod_bases, mut_bases,
                                             delmut_bases, num_reads, files,
                                             qscore_cutoff, nomut_bit, ambig_info,
                                             sur_bases, del_bit, miss_info, bases,
                                             ref_seq, masked_postions)
            else:
                line = next(sam_fileobj)
                line = line.strip().split()
                mate = Mate(line)
                if mate.RNAME == ref_name:
                    GenerateBitVector_Single(mate, refs_seq, phred_qscore,
                                             cov_bases, info_bases, mod_bases,
                                             mut_bases, delmut_bases,
                                             num_reads, files, start, end,
                                             qscore_cutoff, nomut_bit, ambig_info,
                                             sur_bases, del_bit, miss_info, bases,
                                             ref_seq, masked_postions)
        except StopIteration:
            break
    sam_fileobj.close()


class Mate():
    """
    Attributes of a mate in a read pair. Be careful about the order!
    """
    def __init__(self, line_split):
        self.QNAME = line_split[0]
        self.FLAG = line_split[1]
        self.RNAME = line_split[2]
        self.POS = int(line_split[3])
        self.MAPQ = int(line_split[4])
        self.CIGAR = line_split[5]
        self.RNEXT = line_split[6]
        self.PNEXT = int(line_split[7])
        self.TLEN = line_split[8]
        self.SEQ = line_split[9]
        self.QUAL = line_split[10]
        self.MDSTRING = line_split[11].split(":")[2]

    def __repr__(self):
        return self.QNAME + "-" + self.RNAME + "-" + str(self.POS)+"-" + \
            self.CIGAR+"-"+self.MDSTRING


def Convert_Read(mate, refs_seq, phred_qscore,
                 qscore_cutoff, nomut_bit, ambig_info,
                 sur_bases, del_bit, miss_info):
    """
    Convert a read's sequence to a bit vector of 0s & 1s and substituted bases
    Args:
        mate (Mate): Read
        refs_seq (dict): Sequences of the ref genomes in the file
        phred_qscore (dict): Qual score - ASCII symbol mapping
    Returns:
        bitvector_mate (dict): Bitvector. Format: d[pos] = bit
    """
    bitvector_mate = {}  # Mapping of read to 0s and 1s
    read_seq = mate.SEQ  # Sequence of the read
    ref_seq = refs_seq[mate.RNAME]  # Sequence of the ref genome
    q_scores = mate.QUAL  # Qual scores of the bases in the read
    i = mate.POS  # Pos in the ref sequence
    j = 0  # Pos in the read sequence
    CIGAR_Ops = Parse_CIGAR(mate.CIGAR)
    op_index = 0
    while op_index < len(CIGAR_Ops):  # Each CIGAR operation
        op = CIGAR_Ops[op_index]
        desc, length = op[1], int(op[0])

        if desc == 'M':  # Match or mismatch
            for k in range(length):  # Each base
                if phred_qscore[q_scores[j]] >= qscore_cutoff:
                    bitvector_mate[i] = read_seq[j] \
                        if read_seq[j] != ref_seq[i - 1] else nomut_bit
                else:  # < Qscore cutoff
                    bitvector_mate[i] = ambig_info
                i += 1  # Update ref index
                j += 1  # Update read index

        elif desc == 'D':  # Deletion
            for k in range(length - 1):  # All bases except the 3' end
                bitvector_mate[i] = ambig_info
                i += 1  # Update ref index
            ambig = Calc_Ambig_Reads(ref_seq, i, length,
                                                         sur_bases)
            bitvector_mate[i] = ambig_info if ambig else del_bit
            i += 1  # Update ref index

        elif desc == 'I':  # Insertion
            j += length  # Update read index

        elif desc == 'S':  # Soft clipping
            j += length  # Update read index
            if op_index == len(CIGAR_Ops) - 1:  # Soft clipped at the end
                for k in range(length):
                    bitvector_mate[i] = miss_info
                    i += 1  # Update ref index
        else:
            print('Unknown CIGAR op encountered.')
            return ''

        op_index += 1
    return bitvector_mate


def Combine_Mates(bitvector_mate1, bitvector_mate2,
                  bases, nomut_bit, ambig_info):
    """
    Combine bit vectors from mate 1 and mate 2 into a single read's bit vector.
    0 has preference. Ambig info does not. Diff muts in the two mates are
    counted as ambiguous info.
    Args:
        bitvector_mate1 (dict): Bit vector from Mate 1
        bitvector_mate2 (dict): Bit vector from Mate 2
    Returns:
        bit_vector (dict): Bitvector. Format: d[pos] = bit
    """
    bit_vector = {}
    for (pos, bit) in bitvector_mate1.items():  # Bits in mate 1
        bit_vector[pos] = bit
    for (pos, bit) in bitvector_mate2.items():  # Bits in mate2
        if pos not in bitvector_mate1:  # Not present in mate 1
            bit_vector[pos] = bit  # Add to bit vector
        else:  # Overlap in mates
            mate1_bit = bitvector_mate1[pos]
            mate2_bit = bitvector_mate2[pos]
            bits = set([mate1_bit, mate2_bit])
            if len(bits) == 1:  # Both mates have same bit
                bit_vector[pos] = mate1_bit
            else:  # More than one bit
                if nomut_bit in bits:  # 0 in one mate
                    bit_vector[pos] = nomut_bit  # Add 0
                elif ambig_info in bits:  # Ambig info in one mate
                    other_bit = list(bits - set(ambig_info))[0]
                    bit_vector[pos] = other_bit  # Add other bit
                elif mate1_bit in bases and mate2_bit in bases:
                    if mate1_bit != mate2_bit:  # Diff muts on both mates
                        bit_vector[pos] = ambig_info
    return bit_vector


def Plotting_Variables(q_name, ref, bit_vector, start, end, cov_bases,
                       info_bases, mod_bases, mut_bases, delmut_bases,
                       num_reads, files,
                       miss_info, ambig_info, bases, del_bit,
                       ref_seq, masked_postions):
    """
    Create final bit vector in relevant coordinates and all the
    variables needed for plotting
    Args:
        q_name (string): Query name of read
        ref (string): Name of ref genome
        bit_vector (dict): Bit vector from the mate/mates
    """
    # Create bit vector in relevant coordinates
    num_reads[ref] += 1  # Add a read to the count
    bit_string = ''
    for pos in range(start, end + 1):  # Each pos in coords of interest
        if pos not in bit_vector:  # Pos not covered by the read
            read_bit = miss_info
        else:
            if pos in masked_postions:
                info_bases[ref][pos] += 1
                read_bit = '0'
            else:
                read_bit = bit_vector[pos]
                cov_bases[ref][pos] += 1
                if read_bit != ambig_info:
                    info_bases[ref][pos] += 1
                if read_bit in bases:  # Mutation
                    mod_bases[ref][read_bit][pos] += 1
                    mut_bases[ref][pos] += 1
                    delmut_bases[ref][pos] += 1
                elif read_bit == del_bit:  # Deletion
                    delmut_bases[ref][pos] += 1
        bit_string += read_bit
    # Write bit vector to output text file
    n_mutations = str(float(sum(bit.isalpha() for bit in bit_string)))
    if not bit_string.count('.') == len(bit_string):  # Not all '.'
        files[ref].write(q_name + '\t' + bit_string + '\t' + n_mutations + '\n')


def Parse_CIGAR(cigar_string):
    """
    Parse a CIGAR string
    Args:
        cigar_string (string): CIGAR string
    Returns:
        ops (list): List of operations. Each op is of type
        (length, description) such as ('37', 'M'), ('10', 'I'), (24, 'D'), etc.
    """
    import re
    ops = re.findall(r'(\d+)([A-Z]{1})', cigar_string)
    return ops


def Calc_Ambig_Reads(ref_seq, i, length, num_surBases):
    """
    Determines whether a deletion is ambiguous or not by looking at the
    sequence surrounding the deletion. Edge cases not handled right now.
    Args:
        ref_seq (string): Reference sequence
        i (int): 3' index of del at ref sequence
        length (int): Length of deletion
        num_surBases (int): Number of surrounding bases to consider
    Returns:
        boolean: Whether deletion is ambiguous or not
    """
    orig_del_start = i - length + 1
    orig_sur_start = orig_del_start - num_surBases
    orig_sur_end = i + num_surBases
    orig_sur_seq = ref_seq[orig_sur_start - 1: orig_del_start - 1] + \
        ref_seq[i:orig_sur_end]
    for new_del_end in range(i - length, i + length + 1):  # Alt del end points
        if new_del_end == i:  # Orig end point
            continue
        new_del_start = new_del_end - length + 1
        sur_seq = ref_seq[orig_sur_start - 1: new_del_start - 1] + \
            ref_seq[new_del_end:orig_sur_end]
        if sur_seq == orig_sur_seq:
            return True
    return False


def Parse_PhredFile(qscore_filename):
    """
    Parse a file containing Phred Q Score info
    Args:
        qscore_filename (string): Path to Q Score file
    Returns:
        phred_qscore (dict): Mapping of ASCII symbol to Phred Q Score
    """
    phred_qscore = {}
    qscore_file = open(qscore_filename)
    qscore_file.readline()  # Ignore header line
    for line in qscore_file:
        line = line.strip().split()
        score, symbol = int(line[0]), line[1]
        phred_qscore[symbol] = score
    qscore_file.close()
    return phred_qscore


def Run_EMJob(X, bvfile_basename, MIN_ITS, CONV_CUTOFF,
              outplot_dir, K, CPUS, run):

    if K == 1:
        NumReads_File(bvfile_basename, X, outplot_dir)

    EM_res = Run_EM(X, K, MIN_ITS, CONV_CUTOFF, CPUS)
    log_like_list, final_mu, final_obs_pi, final_real_pi, resps, BIC = EM_res

    Run_Plots(bvfile_basename, X, K, log_like_list, final_mu,
                       final_obs_pi, final_real_pi, resps, BIC, outplot_dir,
                       run)

def Run_EM(X, K, MIN_ITS, CONV_CUTOFF, CPUS):
    """
    """
    BETA_A = 1.5  # Beta dist shape parameter
    BETA_B = 20  # Beta dist shape parameter
    conv_string = 'Log like converged after {:d} iterations'
    N, D = X.BV_Matrix.shape[0], X.BV_Matrix.shape[1]

    # Start and end coordinates of matrix for each thread
    calc_inds = calc_matrixIndices(N, K, CPUS)

    # ---------------------- Iterations start ---------------------------- #

    # Initialize DMS modification rate for each base in each cluster
    # by sampling from a beta distribution
    mu = np.asarray([scipy.stats.beta.rvs(BETA_A, BETA_B, size=D)
                    for k in range(K)])

    # Initialize cluster probabilties with uniform distribution
    obs_pi = np.asarray([1.0 / K] * K)

    converged = False
    iteration = 1
    log_like_list, mu_list, obs_pi_list, real_pi_list = [], [], [], []

    while not converged:  # Each iteration of the EM algorithm

        # Expectation step
        (resps, log_like, denom) = Exp_Step(X, K, mu, obs_pi, calc_inds, CPUS)

        # Maximization step
        (mu, obs_pi, real_pi) = Max_Step(X, K, mu, resps, denom)

        log_like_list.append(log_like)
        mu_list.append(mu)
        obs_pi_list.append(obs_pi)
        real_pi_list.append(real_pi)

        # Check if log like has converged
        if iteration >= MIN_ITS:  # At least min iterations has run
            prev_loglike = log_like_list[-2]
            diff = log_like - prev_loglike
            if diff <= CONV_CUTOFF:  # Converged
                converged = True
                print(conv_string.format(iteration))

        iteration += 1

    final_mu = mu_list[-1]
    final_obs_pi, final_real_pi = obs_pi_list[-1], real_pi_list[-1]

    # ------------------------ Iterations end ---------------------------- #

    BIC = calc_BIC(N, X.params_len, K, log_like_list[-1])
    return (log_like_list, final_mu, final_obs_pi, final_real_pi, resps, BIC)


def Exp_Step(X, K, mu, pi, calc_inds, CPUS):
    """
    """
    N, D = X.BV_Matrix.shape[0], X.BV_Matrix.shape[1]
    log_pi = np.log(pi)
    log_pmf = np.zeros((N, D, K))
    denom = [calc_denom(0, mu[k], {}, {}) for k in range(K)]

    input_array1 = [[X, mu, ind, k] for ind in calc_inds for k in range(K)]
    pool1 = ThreadPool(CPUS)
    logpmf_results1 = pool1.starmap(logpmf_function1,
                                    input_array1)
    pool1.close()
    pool1.join()
    for i in range(len(logpmf_results1)):
        ind, k = input_array1[i][2], input_array1[i][3]
        start, end = ind[0], ind[1]
        log_pmf[start:end + 1, :, k] = logpmf_results1[i]

    log_pmf = np.sum(log_pmf, axis=1)  # Sum of log - like taking product

    input_array2 = [[log_pmf, denom, ind, k] for ind in calc_inds
                    for k in range(K)]
    pool2 = ThreadPool(CPUS)
    logpmf_results2 = pool2.starmap(logpmf_function2,
                                    input_array2)
    pool2.close()
    pool2.join()
    for i in range(len(logpmf_results2)):
        ind, k = input_array2[i][2], input_array2[i][3]
        start, end = ind[0], ind[1]
        log_pmf[start:end + 1, k] = logpmf_results2[i]

    log_resps_numer = np.add(log_pi, log_pmf)
    log_resps_denom = scipy.special.logsumexp(log_resps_numer, axis=1)
    log_resps = np.subtract(log_resps_numer.T, log_resps_denom).T
    resps = np.exp(log_resps)

    log_like = np.dot(log_resps_denom, X.BV_Abundance)
    return (resps, log_like, denom)


def Max_Step(X, K, mu, resps, denom):
    """
    """
    D = X.BV_Matrix.shape[1]
    mu, obs_pi, real_pi = np.zeros((K, D)), np.zeros(K), np.zeros(K)
    for k in range(K):
        N_k = np.sum(resps[:, k] * X.BV_Abundance)
        x_bar_k = np.sum((resps[:, k] * X.BV_Abundance *
                          X.BV_Matrix.T).T, axis=0) / N_k
        upd_mu = newton_krylov(lambda mu_k: mu_der(mu_k, x_bar_k),
                               mu[k])
        mu[k] = upd_mu  # Mu with denom correction
        obs_pi[k] = N_k / X.n_bitvectors
    real_pi = [obs_pi[k] / denom[k][0] for k in range(K)]
    real_pi = real_pi / np.sum(real_pi)
    return (mu, obs_pi, real_pi)


def mu_der(mu_k, x_bar_k):
    """
    """
    mu_k_rev = mu_k[::-1]
    denom_k = calc_denom(0, mu_k, {}, {})
    denom_k_rev = calc_denom(0, mu_k_rev, {}, {})
    upd_mu = [(mu_k[i] * denom_k[1][i] * denom_k_rev[1][len(mu_k) - i - 1] /
              denom_k[0]) - x_bar_k[i] for i in range(len(mu_k))]
    return np.array(upd_mu)


def calc_denom(i, mu, denom_probs, s2_probs):
    """
    """
    if i in denom_probs:  # Already encountered
        return (denom_probs[i], s2_probs)
    elif i >= len(mu):  # Base case
        return (1, s2_probs)
    else:  # Make the calc
        s1 = calc_denom(i + 1, mu, denom_probs, s2_probs)[0]
        s2 = (1.0 - mu[i + 1: i + 4]).prod() * \
            calc_denom(i + 4, mu, denom_probs, s2_probs)[0]
        denom_probs[i] = ((1 - mu[i]) * s1) + (mu[i] * s2)
        s2_probs[i] = s2
        return (denom_probs[i], s2_probs)


def is_distmuts_valid(bs):
    """
    """
    for i in range(len(bs)):
        if bs[i] == '1':
            try:
                if i - latest_mutbit_index < 4:
                    return False
            except NameError:  # This happens the first time we see a '1'
                None
            latest_mutbit_index = i
    return True


def is_surmuts_valid(bs):
    """
    """
    invalid_set = ['.1', '?1', '1.', '1?']
    for i in range(len(bs)):
        if bs[i:i + 2] in invalid_set:
            return False
    return True


def calc_nmuts_thresh(bv_filename):
    """
    """
    n_muts = pd.read_csv(bv_filename, sep='\t', skiprows=2,
                         usecols=['N_Mutations'], index_col=False)
    n_muts = n_muts['N_Mutations']
    mad = abs(n_muts - n_muts.median()).median()
    nmuts_thresh = n_muts.median() + (3 * mad / 0.6745)
    return int(round(nmuts_thresh))


def logpmf_function1(X, mu, ind, k):
    """
    """
    start, end = ind[0], ind[1]
    return scipy.stats.bernoulli.logpmf(X.BV_Matrix[start:end + 1], mu[k])


def logpmf_function2(log_pmf, denom, ind, k):
    """
    """
    start, end = ind[0], ind[1]
    return log_pmf[start:end + 1, k] - math.log(denom[k][0])


def calc_matrixIndices(N, K, cpus):
    """
    """
    calcsPerCPU = max(round(N * K / cpus), 1)
    inds, start = [], 0
    while start < N:
        coord = (start, start + calcsPerCPU - 1)
        inds.append(coord)
        start = start + calcsPerCPU
    return inds


def calc_BIC(N, PARAMS_LEN, K, log_like):
    """
    """
    return math.log(N) * PARAMS_LEN * K - (2 * log_like)


def Run_Plots(sample_name, X, K, log_like_list, final_mu, final_obs_pi,
              final_real_pi, resps, BIC, outplots_dir, run):
    """
    """
    K_dir = outplots_dir + '/K_' + str(K) + '/'
    if not os.path.exists(K_dir):
        os.makedirs(K_dir)
    run_dir = K_dir + 'run_' + str(run) + '/'
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    indices = X.indices.split(',')
    start, end = int(indices[0]), int(indices[1])
    seq = X.seq

    # File 1 - List of log likelihoods
    outfile_name1 = run_dir + 'Log_Likelihoods.txt'
    outfile1 = open(outfile_name1, 'w')
    for log_like in log_like_list:
        str_loglike = str(round(log_like, 2))
        outfile1.write(str_loglike + '\n')
    outfile1.close()

    # File 2 - Largest log likelihood
    outfile_name2 = run_dir + 'Largest_LogLike.txt'
    outfile2 = open(outfile_name2, 'w')
    str_loglike = str(round(log_like_list[-1], 2))
    outfile2.write(str_loglike + '\n')
    outfile2.close()

    # File 3 - BIC
    outfile_name3 = run_dir + 'BIC.txt'
    outfile3 = open(outfile_name3, 'w')
    str_BIC = str(round(BIC, 2))
    outfile3.write(str_BIC + '\n')
    outfile3.close()

    # File 4 - Cluster mus
    outfile_name4 = run_dir + 'Clusters_Mu.txt'
    outfile4 = open(outfile_name4, 'w')
    outfile4.write('@ref' + '\t' + X.ref_file + ';' + X.ref + '\t' +
                   seq[start - 1:end] + '\n')
    outfile4.write('@coordinates:length' + '\t' + str(start) + ',' +
                   str(end) + ':' + str(end - start + 1) + '\n')
    outfile4.write('Position')
    for i in range(len(final_mu)):
        outfile4.write('\tCluster_' + str(i + 1))
    outfile4.write('\n')
    for i in range(start, end + 1):
        outfile4.write(str(i))
        for j in range(len(final_mu)):
            outfile4.write('\t' + str(round(final_mu[j][i-start], 5)))
        outfile4.write('\n')
    outfile4.close()

    # File 5 - responsibilities
    outfile_name5 = run_dir + 'Responsibilities.txt'
    outfile5 = open(outfile_name5, 'w')
    outfile5.write('Number\t')
    for k in range(K):
        k += 1
        outfile5.write('Cluster_' + str(k) + '\t')
    outfile5.write('N\tBit_vector\n')
    index_num = 1
    for bit_vect in X.n_occur:
        bv = ''.join(bit_vect)
        abundance = str(X.n_occur[bit_vect])
        outfile5.write(str(index_num) + '\t')
        for k in range(K):
            outfile5.write(str(round(resps[index_num-1][k], 3)) + '\t')
        outfile5.write(abundance + '\t' + bv + '\n')
        outfile5.write('\n')
        index_num += 1
    outfile5.close()

    # File 6 - Cluster proportions
    outfile_name6 = run_dir + 'Proportions.txt'
    outfile6 = open(outfile_name6, 'w')
    outfile6.write('Cluster, Obs Pi, Real pi \n')
    for k in range(K):
        obs_prob = str(round(final_obs_pi[k], 2))
        real_prob = str(round(final_real_pi[k], 2))
        outfile6.write(str(k+1) + ',' + obs_prob + ',' + real_prob + '\n')
    outfile6.close()

    # Plot 1 - log likelihood vs iteration number
    loglike_trace = go.Scatter(
        x=[(i+1) for i in range(len(log_like_list))],
        y=log_like_list,
        mode='lines'
    )
    loglike_layout = dict(xaxis=dict(title='Iteration'),
                          yaxis=dict(title='Log likelihood'))
    loglike_data = [loglike_trace]
    loglike_fig = dict(data=loglike_data, layout=loglike_layout)
    plotly.offline.plot(loglike_fig, filename=run_dir +
                        'LogLikes_Iterations.html',
                        auto_open=False)

    # Plot 2 - DMS mod rate for each base in each cluster
    DMSModRate_cluster_data = []
    xaxis_coords = [i for i in range(start, end+1)]
    for k in range(K):
        obs_prob = round(final_obs_pi[k], 2)
        real_prob = round(final_real_pi[k], 2)
        c_name = 'Cluster ' + str(k + 1) + ', obs p=' + str(obs_prob) + \
                 ', real p=' + str(real_prob)
        trace = go.Scatter(
            x=xaxis_coords,
            y=final_mu[k],
            name=c_name,
            mode='lines+markers'
        )
        DMSModRate_cluster_data.append(trace)
    DMSModRate_cluster_layout = dict(xaxis=dict(title='Position (BP)'),
                                     yaxis=dict(title='DMS mod rate'))
    DMSModRate_cluster_fig = dict(data=DMSModRate_cluster_data,
                                  layout=DMSModRate_cluster_layout)
    plotly.offline.plot(DMSModRate_cluster_fig, filename=run_dir +
                        'DMSModRate.html',
                        auto_open=False)

    # Plot 3 - Same as Plot 2, but in subplots
    cmap = {'A': 'red', 'T': 'green', 'G': 'orange', 'C': 'blue'}  # Color map
    colors = [cmap[seq[i]] for i in range(len(seq))]
    ref_bases = [seq[i] for i in range(len(seq))]
    titles = ['Cluster ' + str(k+1) for k in range(K)]
    fig3 = tools.make_subplots(rows=K, cols=1, subplot_titles=titles)
    for k in range(K):
        trace = go.Bar(
            x=xaxis_coords,
            y=final_mu[k],
            text=ref_bases,
            marker=dict(color=colors),
            showlegend=False
        )
        fig3.append_trace(trace, k + 1, 1)
    plotly.offline.plot(fig3, filename=run_dir +
                        'DMSModRate_Clusters.html', auto_open=False)


def NumReads_File(sample_name, X, outplots_dir):
    """
    """
    outfile_name = outplots_dir + 'BitVectors_Filter.txt'
    outfile = open(outfile_name, 'w')
    outfile.write('Number of bit vectors used: ' + str(X.n_bitvectors) + '\n')
    outfile.write('Number of unique bit vectors used: ' +
                  str(X.n_unique_bitvectors) + '\n')
    outfile.write('Number of bit vectors discarded: ' +
                  str(X.n_discard) + '\n')
    outfile.close()


def Load_BitVectors(bv_file, INFO_THRESH, SIG_THRESH, inc_TG, output_dir):
    """
    """
    bases = ['A', 'T', 'G', 'C']
    bit_strings, mut_popavg, n_discard = [], {}, 0
    f, f1, f2, f3, f4 = 0, 0, 0, 0, 0

    bv_fileobj = open(bv_file)
    bvfile_contents = bv_fileobj.readlines()
    bv_fileobj.close()

    first_line = bvfile_contents[0]
    first_line_split = first_line.strip().split()
    ref_info, seq = first_line_split[1], first_line_split[2]
    ref_file, ref = ref_info.split(';')[0], ref_info.split(';')[1]

    second_line = bvfile_contents[1]
    second_line_split = second_line.strip().split()
    indices = second_line_split[1].split(':')[0]

    l = len(bvfile_contents[3].strip().split()[1])  # Len of 1st bit string
    nmuts_min = int(round(0.1 * l))
    nmuts_thresh = max(nmuts_min, calc_nmuts_thresh(bv_file))
    # print('Mutations threshold:', nmuts_thresh)

    for i in range(3, len(bvfile_contents)):
        f += 1
        line = bvfile_contents[i].strip().split()
        bit_string = line[1]
        n_mut = float(line[2])

        # Replace bases with 1
        for base in bases:
            bit_string = bit_string.replace(base, '1')

        # Filter 1 - Number of mutations
        if n_mut > nmuts_thresh:
            n_discard += 1
            f1 += 1
            continue

        # Filter 2 - Fraction of informative bits
        if (bit_string.count('.') + bit_string.count('?') +
           bit_string.count('N')) >= INFO_THRESH * len(bit_string):
            n_discard += 1
            f2 += 1
            continue

        # Filter 3 - Distance between mutations
        if not is_distmuts_valid(bit_string):
            n_discard += 1
            f3 += 1
            continue

        # Filter 4 - Bits surrounding mutations
        if not is_surmuts_valid(bit_string):
            n_discard += 1
            f4 += 1
            continue

        bit_strings.append(bit_string)

    """
    print('Total bit vectors:', f)
    print('Bit vectors removed because of too many mutations: ', f1)
    print('Bit vectors removed because of too few informative bits: ', f2)
    print('Bit vectors removed because of mutations close by: ', f3)
    print('Bit vectors removed because of no info around mutations: ', f4)
    """

    D = len(bit_strings[0])
    thresh_pos = []  # Positions below signal threshold
    noparam_pos = set()  # Positions for which no param is estimated
    for d in range(D):  # Each position of interest in the genome
        bits_list = [bs[d] for bs in bit_strings]  # List of bits at that pos
        noinfo_count = bits_list.count('.') + bits_list.count('?') + \
            bits_list.count('N')
        info_count = len(bits_list) - noinfo_count  # Num of informative bits
        try:
            mut_prob = bits_list.count('1') / info_count
        except ZeroDivisionError:
            mut_prob = 0
        if mut_prob < SIG_THRESH:
            mut_prob = 0
            thresh_pos.append(d)
            noparam_pos.add(d)
        mut_popavg[d] = mut_prob

    for i in range(len(bit_strings)):  # Change . and ? to 0, noise to 0
        bit_string = bit_strings[i]
        bit_string = bit_string.replace('?', '0')
        bit_string = bit_string.replace('.', '0')
        bit_string = bit_string.replace('N', '0')

        # Suppressing data from Ts and Gs
        if not inc_TG:
            bit_string = list(bit_string)
            j = 0
            while j < len(bit_string):
                if seq[j] == 'T' or seq[j] == 'G':
                    noparam_pos.add(j)
                    bit_string[j] = '0'
                j += 1
            bit_string = ''.join(bit_string)

        bit_string = np.array(list(bit_string))
        bit_string[thresh_pos] = '0'
        bit_string = ''.join(bit_string)

        bit_strings[i] = bit_string

    params_len = D - len(noparam_pos)
    X = BV_Object(bit_strings, mut_popavg, n_discard, ref_file,
                           ref, seq, output_dir, indices, params_len)
    return X


class BV_Object():
    """
    """
    def __init__(self, bit_vectors, mut_popavg, n_discard, ref_file,
                 ref, seq, infiles_dir, indices, params_len):
        BV_Matrix, BV_Abundance, n_occur = [], [], {}
        for bit_vector in bit_vectors:
            bit_vector = tuple(bit_vector)  # Change to a tuple
            if bit_vector in n_occur:
                n_occur[bit_vector] += 1
            else:
                n_occur[bit_vector] = 1
        for bit_vector in n_occur:
            bv = np.array(list(map(float, bit_vector)))  # Convert to float
            BV_Matrix.append(bv)
            BV_Abundance.append(n_occur[bit_vector])

        BV_Matrix = np.array(BV_Matrix)
        BV_Abundance = np.array(BV_Abundance)
        self.BV_Matrix = BV_Matrix  # Only unique bit vectors
        self.BV_Abundance = BV_Abundance  # Abundance of each bit vector
        self.n_occur = n_occur
        self.n_bitvectors = len(bit_vectors)
        self.n_unique_bitvectors = len(n_occur.keys())
        self.n_discard = n_discard
        self.mut_popavg = mut_popavg
        self.ref = ref
        self.ref_file = ref_file
        self.seq = seq
        self.infiles_dir = infiles_dir
        self.indices = indices
        self.params_len = params_len


def Collect_BestBIC(sample_name, K, outfiles_dir):
    """
    """
    K_dir = outfiles_dir + 'K_' + str(K) + '/'
    loglikes_file_name = K_dir + 'log_likelihoods.txt'
    loglikes_file = open(loglikes_file_name)
    for line in loglikes_file:
        line_split = line.strip().split()
        run_info = line_split[0]
        if run_info[-4:] == 'best':  # BIC from best run
            BIC = float(line_split[2])
            return BIC


def Post_Process(sample_name, ref_file, K, RUNS, cur_BIC, norm_bases,
                 struct, input_dir, outfiles_dir):
    """
    """
    largest_loglike, BICs, log_likes, best_run = float('-inf'), [], [], ''

    for run in range(1, RUNS + 1):
        run_dir = outfiles_dir + '/K_' + str(K) + '/' + \
                      'run_' + str(run) + '/'

        largest_loglikefilename = run_dir + 'Largest_LogLike.txt'
        largest_loglikefile = open(largest_loglikefilename)
        log_like = float(largest_loglikefile.readline())
        largest_loglikefile.close()

        BIC_filename = run_dir + 'BIC.txt'
        BIC_file = open(BIC_filename)
        BIC = float(BIC_file.readline())
        BIC_file.close()

        log_likes.append(log_like)
        BICs.append(BIC)

        if log_like > largest_loglike:
            largest_loglike = log_like
            best_run = run

        os.system('rm ' + run_dir + 'Largest_LogLike.txt')
        os.system('rm ' + run_dir + 'BIC.txt')

    # Write to log likelihoods file
    LogLikes_File(sample_name, K, RUNS, log_likes, BICs,
                           best_run, outfiles_dir)

    # Rename directory of best run
    orig_dir = outfiles_dir + 'K_' + str(K) + '/' + \
        'run_' + str(best_run) + '/'
    new_dir = outfiles_dir + 'K_' + str(K) + '/' + \
        'run_' + str(best_run) + '-best/'
    os.system('mv ' + orig_dir + ' ' + new_dir)

    clustmu_file = new_dir + 'Clusters_Mu.txt'

    # Folding with RNAstructure
    if struct:
        # Num bases on each side of region for secondary structure prediction
        num_bases = [0, 50, 100, 150, 200]
        for num_base in num_bases:
            ConstraintFoldDraw(ref_file, clustmu_file,
                                num_base, num_base, norm_bases)

    # Scatter plot of reactivities
    if K > 1:
        Scatter_Clusters(ref_file, clustmu_file)


def LogLikes_File(sample_name, K, RUNS, log_likes, BICs,
                  best_run, outplots_dir):
    """
    """
    K_dir = outplots_dir + '/K_' + str(K) + '/'
    loglikes_file_name = K_dir + 'log_likelihoods.txt'
    loglikes_file = open(loglikes_file_name, 'w')
    loglikes_file.write('Run\tLog_likelihood\tBIC_score\n')
    for run in range(1, RUNS + 1):
        log_like = str(round(log_likes[run - 1], 2))
        BIC = str(round(BICs[run - 1], 2))
        line = str(run) + '\t' + log_like + '\t' + BIC + '\n'
        if run == best_run:
            line = str(run)+'-best' + '\t' + log_like + '\t' + BIC + '\n'
        loglikes_file.write(line)
    loglikes_file.close()


def ConstraintFoldDraw(ref_filename, clustMuFile, expUp, expDown, norm_bases):
    """
    """
    sample = clustMuFile.split('/')[-4]
    file_dir = os.path.dirname(clustMuFile)

    # Create trimmed reference fasta file
    clustMuFileContents = open(clustMuFile)
    first_line = clustMuFileContents.readline().strip()
    second_line = clustMuFileContents.readline().strip()
    clustMuFileContents.close()
    first_line_split = first_line.strip().split()
    ref_info = first_line_split[1]
    ref_file, ref = ref_info.split(';')[0], ref_info.split(';')[1]
    #ref_file = input_dir + ref_file + '.fasta'
    #refs_seq = Parse_FastaFile(ref_file)
    refs_seq = Parse_FastaFile(ref_filename)
    entire_seq = refs_seq[ref]
    second_line_split = second_line.strip().split()
    indices = second_line_split[1].split(':')[0]
    start, end = int(indices.split(',')[0]), int(indices.split(',')[1])
    mod_start = max(1, start - expUp)
    mod_end = min(end + expDown, len(entire_seq))
    trim_seq = entire_seq[mod_start - 1:mod_end]
    trimref_filename = file_dir + '/' + ref + '_trimUp_' + str(expUp) + \
        '_trimDown_' + str(expDown) + '.fa'
    ref_name = ref + '_' + str(mod_start) + '_' + str(mod_end)
    Create_FastaFile(trimref_filename, ref_name, trim_seq)

    # Gather mus for every k and normalize them
    clusts_mus = pd.read_csv(clustMuFile, sep='\t', skiprows=2,
                             index_col=False)
    rows, K = len(clusts_mus), len(clusts_mus.columns) - 1
    norm_clusts_mus = np.zeros((rows, K))
    for k in range(K):
        mus = clusts_mus['Cluster_' + str(k + 1)]
        norm_value = med(np.sort(mus)[-1:-(norm_bases+1):-1])  # Median of mus
        norm_mus = mus / norm_value  # Normalize the mus
        norm_mus[norm_mus > 1.0] = 1.0  # Cap at 1
        norm_clusts_mus[:, k] = norm_mus
    norm_clusts_mus = np.around(norm_clusts_mus, decimals=3)

    # Drawing of structure for each k
    for k in range(K):
        clust_name = file_dir + '/' + sample + '-K' + str(K) + '_Cluster' + str(k+1)

        const_filename = clust_name + '_expUp_' + str(expUp) + '_expDown_' + \
            str(expDown) + '_const.txt'
        const_file = open(const_filename, 'w')
        for i in range(len(clusts_mus)):
            pos = clusts_mus['Position'][i]
            mod_pos = pos - mod_start + 1  # Pos wrt trimmed seq
            mu = str(norm_clusts_mus[i][k])
            if mu == 'nan':  # Happens in UT
                mu = '0'
            if entire_seq[pos-1] == 'T' or entire_seq[pos-1] == 'G':
                mu = '-999'
            if mod_pos > 0 and mod_start <= pos <= mod_end:  # Can be < 0 because its wrt trimmed seq
                const_file.write(str(mod_pos) + '\t' + mu + '\n')
        const_file.close()

        # Folding using RNAstructure
        ct_filename = clust_name + '_expUp_' + str(expUp) + '_expDown_' + \
            str(expDown) + '.ct'
        dot_filename = clust_name + '_expUp_' + str(expUp) + '_expDown_' + \
            str(expDown) + '.dot'
        pic_filename = clust_name + '_basesExpanded_' + str(expUp) + '.ps'
        fold_command = '/rumi/shams/khorms/programs/RNAstructure/exe/Fold -m 3 ' + trimref_filename + ' -dms ' + \
            const_filename + ' ' + ct_filename
        ct2dot_command = '/rumi/shams/khorms/programs/RNAstructure/exe/ct2dot ' + ct_filename + ' ALL ' + \
            dot_filename
        draw_command = '/rumi/shams/khorms/programs/RNAstructure/exe/draw ' + dot_filename + ' ' + \
            pic_filename + ' -S ' + const_filename
        add_database_command = "export DATAPATH=/rumi/shams/khorms/programs/RNAstructure/data_tables"
        full_command = "%s ; %s ; %s ; %s" % (add_database_command, fold_command,
                                              ct2dot_command, draw_command)
        os.system(full_command)
        # os.system(fold_command)
        # os.system(ct2dot_command)
        # os.system(draw_command)

        # Delete unnecessary files
        os.system('rm ' + const_filename)
        os.system('rm ' + ct_filename)
        os.system('rm ' + dot_filename)

    os.system('rm ' + trimref_filename)


def Scatter_Clusters(ref_filename, clustMuFile):
    """
    """
    file_dir = os.path.dirname(clustMuFile)
    clustMuFileContents = open(clustMuFile)
    first_line = clustMuFileContents.readline().strip()
    clustMuFileContents.close()
    first_line_split = first_line.strip().split()
    ref_info = first_line_split[1]
    ref_file, ref = ref_info.split(';')[0], ref_info.split(';')[1]
    #ref_file = input_dir + ref_file + '.fasta'
    #refs_seq = Parse_FastaFile(ref_file)
    refs_seq = Parse_FastaFile(ref_filename)
    entire_seq = refs_seq[ref]
    clusts_mus = pd.read_csv(clustMuFile, sep='\t', skiprows=2,
                             index_col=False)
    positions = clusts_mus['Position']
    valid_indices1 = [i for i in range(len(positions)) if
                      entire_seq[positions[i] - 1] == 'A' or
                      entire_seq[positions[i] - 1] == 'C']  # No Ts and Gs
    K = len(clusts_mus.columns) - 1
    r2_filename = file_dir + '/NormMu_RSquare.txt'
    r2_file = open(r2_filename, 'w')
    r2_file.write('ClusterA-ClusterB:R-Square,P-value\n')
    for k1 in range(K-1):
        for k2 in range(k1 + 1, K):
            mus1 = clusts_mus['Cluster_' + str(k1 + 1)]
            mus2 = clusts_mus['Cluster_' + str(k2 + 1)]
            i1 = np.where(mus1 > 0)[0]  # Non zero indices
            i2 = np.where(mus2 > 0)[0]  # Non zero indices
            valid_indices2 = [i for i in i1 if i in i2]  # Present in both

            mus1 = [mus1[i] for i in range(len(mus1)) if i in valid_indices1 and
                    i in valid_indices2]
            mus2 = [mus2[i] for i in range(len(mus2)) if i in valid_indices1 and
                    i in valid_indices2]
            if len(mus1) == 0 or len(mus2) == 0:  # Like in UT
                continue

            norm_value1 = np.sort(mus1)[-1]
            norm_mus1 = mus1 / norm_value1
            norm_value2 = np.sort(mus2)[-1]
            norm_mus2 = mus2 / norm_value2

            # Plot 1 - normalized mus
            slope1, intercept1, r1, p1, std_err1 = linregress(norm_mus1,
                                                              norm_mus2)
            r1 = r1**2
            r1, p1 = round(r1, 2), round(p1, 2)
            trace1 = go.Scatter(
                x=norm_mus1,
                y=norm_mus2,
                mode='markers',
                marker=dict(color='rgb(49,130,189)'),
                showlegend=False
            )
            trace2 = go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='rgb(49,130,189)'),
                showlegend=False
            )
            title1 = 'R-squared: ' + str(r1) + ', p-value: ' + str(p1)
            layout1 = dict(title=title1,
                           xaxis=dict(title='Cluster '+str(k1 + 1)),
                           yaxis=dict(title='Cluster '+str(k2 + 1)))
            data1 = [trace1, trace2]
            fig1 = dict(data=data1, layout=layout1)
            fname1 = file_dir + '/Cluster' + str(k1 + 1) + '_Cluster' + \
                str(k2 + 1) + '_normmus.html'
            plotly.offline.plot(fig1, filename=fname1, auto_open=False)

            # Plot 2 - mus
            slope2, intercept2, r2, p2, std_err2 = linregress(mus1, mus2)
            r2 = r2 ** 2
            r2, p2 = round(r2, 2), round(p2, 2)
            m1 = max(max(mus1), max(mus2))
            trace1_1 = go.Scatter(
                x=mus1,
                y=mus2,
                mode='markers',
                marker=dict(color='rgb(49,130,189)'),
                showlegend=False
            )
            trace2_1 = go.Scatter(
                x=[0, m1],
                y=[0, m1],
                mode='lines',
                line=dict(color='rgb(49,130,189)'),
                showlegend=False
            )
            title2 = 'R-squared: ' + str(r2) + ', p-value: ' + str(p2)
            layout2 = dict(title=title2,
                           xaxis=dict(title='Cluster '+str(k1 + 1)),
                           yaxis=dict(title='Cluster '+str(k2 + 1)))
            data2 = [trace1_1, trace2_1]
            fig2 = dict(data=data2, layout=layout2)
            fname2 = file_dir + '/Cluster' + str(k1 + 1) + '_Cluster' + \
                str(k2 + 1) + '_mus.html'
            plotly.offline.plot(fig2, filename=fname2, auto_open=False)

            r2_file.write('Cluster' + str(k1+1) + '-Cluster' + str(k2+1) +
                          ':' + str(r1) + ',' + str(p1) + '\n')
    r2_file.close()


def Log_File(sample_name, NUM_RUNS, MIN_ITS,
             CONV_CUTOFF, INFO_THRESH, SIG_THRESH, INC_TG,
             norm_bases, K, time_taken, outplots_dir):
    """
    """
    now = datetime.datetime.now()
    log_file_name = outplots_dir + 'log.txt'
    log_file = open(log_file_name, 'w')
    log_file.write('Sample: ' + sample_name + '\n')
    log_file.write('Number of EM runs: ' + str(NUM_RUNS) + '\n')
    log_file.write('Minimum number of iterations: ' + str(MIN_ITS) + '\n')
    log_file.write('Convergence cutoff: ' + str(CONV_CUTOFF) + '\n')
    log_file.write('Informative bits threshold: ' + str(INFO_THRESH) + '\n')
    log_file.write('Signal threshold: ' + str(SIG_THRESH) + '\n')
    log_file.write('Include Ts and Gs?: %r\n' % INC_TG)
    log_file.write('Num bases for normalization: ' + str(norm_bases) + '\n')
    log_file.write('Predicted number of clusters: ' + str(K) + '\n')
    log_file.write('Time taken: ' + str(time_taken) + ' mins\n')
    log_file.write('Finished at: ' + now.strftime("%Y-%m-%d %H:%M") + '\n')
    log_file.close()