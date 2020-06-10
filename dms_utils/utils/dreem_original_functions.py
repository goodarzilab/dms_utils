



def GenerateBitVector_Paired(mate1, mate2, refs_seq, phred_qscore,
                             cov_bases, info_bases, mod_bases, mut_bases,
                             delmut_bases, num_reads, files):
    """
    Create a bitvector for paired end sequencing.
    """
    bitvector_mate1 = Convert_Read(mate1, refs_seq, phred_qscore,
                                   QSCORE_CUTOFF, nomut_bit, ambig_info,
                                   SUR_BASES, del_bit, miss_info)
    bitvector_mate2 = Convert_Read(mate2, refs_seq, phred_qscore,
                                   QSCORE_CUTOFF, nomut_bit, ambig_info,
                                   SUR_BASES, del_bit, miss_info)
    bit_vector = Combine_Mates(bitvector_mate1, bitvector_mate2)
    Plotting_Variables(mate1.QNAME, mate1.RNAME, bit_vector, start, end,
                       cov_bases, info_bases, mod_bases, mut_bases,
                       delmut_bases, num_reads, files)


def Process_SamFile(sam_file, paired, refs_seq, start, end,
                    cov_bases, info_bases, mod_bases,
                    mut_bases, delmut_bases, num_reads, files,
                    ref_name, phred_qscore):
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
                    GenerateBitVector_Paired(mate1, mate2, refs_seq,
                                             phred_qscore, cov_bases,
                                             info_bases, mod_bases, mut_bases,
                                             delmut_bases, num_reads, files)
            else:
                line = next(sam_fileobj)
                line = line.strip().split()
                mate = Mate(line)
                if mate.RNAME == ref_name:
                    GenerateBitVector_Single(mate, refs_seq, phred_qscore,
                                             cov_bases, info_bases, mod_bases,
                                             mut_bases, delmut_bases,
                                             num_reads, files, start, end)
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
                 QSCORE_CUTOFF, nomut_bit, ambig_info,
                 SUR_BASES, del_bit, miss_info):
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
                if phred_qscore[q_scores[j]] >= QSCORE_CUTOFF:
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
                                                         SUR_BASES)
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


def Combine_Mates(bitvector_mate1, bitvector_mate2):
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
                       num_reads, files):
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


def GenerateBitVector_Single(mate, refs_seq, phred_qscore,
                             cov_bases, info_bases, mod_bases,
                             mut_bases, delmut_bases, num_reads, files,
                             start, end):
    """
    Create a bitvector for single end sequencing.
    """
    bit_vector = Convert_Read(mate, refs_seq, phred_qscore,
                 QSCORE_CUTOFF, nomut_bit, ambig_info,
                 SUR_BASES, del_bit, miss_info)
    Plotting_Variables(mate.QNAME, mate.RNAME, bit_vector, start, end,
                       cov_bases, info_bases, mod_bases, mut_bases,
                       delmut_bases, num_reads, files)


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