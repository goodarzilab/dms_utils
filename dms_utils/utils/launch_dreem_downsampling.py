import sys
sys.path.append('/rumi/shams/khorms/programs/dms_utils')
import os
import argparse
import multiprocessing

import dms_utils.utils.dreem_utils as dreem_utils
import dms_utils.utils.dreem_original_functions as dof

def handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="", type=str)
    parser.add_argument("--output_folder", help="", type=str)
    parser.add_argument("--n_processes", help="", type=int)

    parser.add_argument("--reference_fasta_filename", help="", type=str)
    parser.add_argument("--reference_name", help="", type=str)
    parser.add_argument("--phred_ascii", help="", type=str)
    parser.add_argument("--masked_postions",help="", type=int)
    parser.add_argument("--start", help="", type=int)
    parser.add_argument("--end", help="", type=int)
    parser.add_argument("--num_runs", help="", type=int)
    parser.add_argument("--min_its", help="", type=int)
    parser.add_argument("--conv_cutoff", help="", type=int)
    parser.add_argument("--cpus", help="", type=int)
    parser.add_argument("--paired", help="", type=bool)

    parser.set_defaults(
        input_folder = '/rumi/shams/khorms/projects/SNIP_switchers/published_DMSseq_data/tomezsko_2020/stemsAC/StemA_C/downsampling/06_04_20_sam',
        output_folder = '/rumi/shams/khorms/projects/SNIP_switchers/published_DMSseq_data/tomezsko_2020/stemsAC/StemA_C/downsampling/06_04_20_dreem',
        n_processes = 15,

        reference_fasta_filename="/rumi/shams/khorms/projects/SNIP_switchers/published_DMSseq_data/tomezsko_2020/stemsAC/StemA_C/reference/MRPS21_200_nt.fa",
        reference_name = "MRPS21_200nt",
        phred_ascii = "/rumi/shams/khorms/programs/dreem/phred_ascii.txt",
        masked_postions = 101,
        start = 80,
        end = 135,
        paired = False,
        num_runs = 10,
        min_its = 300,
        conv_cutoff = 1,
        cpus = 2,
    )
    args = parser.parse_args()
    return args

def declare_global_variables(args):
    global reference_fasta_filename
    global reference_name
    global phred_ascii
    global masked_postions
    global start
    global end
    global paired
    global num_runs
    global min_its
    global conv_cutoff
    global cpus

    reference_fasta_filename = args.reference_fasta_filename
    reference_name = args.reference_name
    phred_ascii = args.phred_ascii
    masked_postions = [args.masked_postions]
    start = args.start
    end = args.end
    paired = args.paired
    num_runs = args.num_runs
    min_its = args.min_its
    conv_cutoff = args.conv_cutoff
    cpus = args.cpus



def make_an_iterable_for_multiprocessing(inp_folder, out_folder):
    iterator_list_of_inputs = []
    for fn in os.listdir(inp_folder):
        if not fn.endswith(".sam"):
            continue
        inp_filename = os.path.join(inp_folder, fn)
        sample_name = fn.replace(".sam","")
        curr_tuple = (inp_filename, sample_name, out_folder)
        iterator_list_of_inputs.append(curr_tuple)
    return iterator_list_of_inputs



def worker_for_parallel_implementation(inp_tuple):
    inp_filename, sample_name, out_folder = inp_tuple
    print("Launching %s" % sample_name)
    dreem_utils.launch_something(sample_name=sample_name,
                                 ref_file=reference_fasta_filename,
                                 ref_name=reference_name,
                                 sam_file=inp_filename,
                                 out_folder=out_folder,
                                 qscore_file=phred_ascii,
                                 masked_postions=masked_postions,
                                 start=start, end=end,
                                 paired=paired,
                                 NUM_RUNS=num_runs,
                                 MIN_ITS=min_its,
                                 CONV_CUTOFF=conv_cutoff,
                                 CPUS=2
                                 )
    print("Calculated %s" % sample_name)


def launch_processes(n_processes, inp_folder, out_folder):
    pool = multiprocessing.Pool(n_processes)
    iterator_list_of_inputs = make_an_iterable_for_multiprocessing(inp_folder, out_folder)
    pool.map(worker_for_parallel_implementation, iterator_list_of_inputs)




def main():
    args = handler()
    declare_global_variables(args)
    launch_processes(args.n_processes, args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()