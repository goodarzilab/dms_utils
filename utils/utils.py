import pysam
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import scipy.stats

def report_reads_statistics(inp_filename,
                            stop_after_N_elements=-1,
                            plot_hist=True,
                            ):
    tic = time.time()
    bam_loc = pysam.AlignmentFile(inp_filename, "rb")
    header_dict_loc = bam_loc.header.to_dict()['SQ']
    ids_dict = get_contig_indices(header_dict_loc)
    n_elements = len(ids_dict)

    print("Total number of elements: ", n_elements)

    reads_per_element = np.zeros(n_elements)
    number_reads_categories = {"total": 0,
                               "passed": 0,
                               "qcfail": 0,
                               "unmapped": 0,
                               "multimapped": 0,
                               "insertions": 0,
                               "deletions": 0,
                               "skipped_regions": 0}

    el_counter = 0

    for header_el in header_dict_loc:
        el_name = header_el['SN']
        el_length = header_el['LN']
        el_number = ids_dict[el_name]

        reads_per_element[el_number] = bam_loc.count(el_name, read_callback='all')

        el_counter += 1
        if stop_after_N_elements > 0:
            if el_counter > stop_after_N_elements:
                break

    toc = time.time()
    print("Calculated in: %d" % (toc - tic))

    if plot_hist:
        plt.hist(reads_per_element, bins=50)
        plt.title("Number of reads per element")
        plt.show()


# if __name__ == "__main__":
#     main()
