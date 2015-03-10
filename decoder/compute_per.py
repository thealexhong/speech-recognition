# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

# Borrows from Kaldi C++ implementation.
# No need to modify file.
import argparse, time, os, sys
from copy import copy


class error_stats(object):
    def __init__(self, ins, dels, subs, tot):
        self.ins, self.dels, self.subs, self.total_cost = \
                                  ins, dels, subs, tot

def LevenshteinEditDistance(ref_trans, hyp_trans):
    # temp sequence to remember error type and stats.
    e, cur_e = [], []
    lst_ref = ref_trans.split(" ")
    for i in range(len(lst_ref)+1):
        e.append(error_stats(0, i, 0, i))
        cur_e.append(error_stats(0, 0, 0, 0))

    lst_hyp = hyp_trans.split(" ")
    for hyp_index in range(1, len(lst_hyp)+1):
        cur_e[0] = copy(e[0])
        cur_e[0].ins += 1
        cur_e[0].total_cost += 1

        for ref_index in range(1, len(lst_ref) +1):
            ins_err = e[ref_index].total_cost + 1
            del_err = cur_e[ref_index-1].total_cost + 1
            sub_err = e[ref_index-1].total_cost
            if lst_hyp[hyp_index-1] != lst_ref[ref_index-1]:
                sub_err += 1

            if sub_err < ins_err and sub_err < del_err:
                cur_e[ref_index] = copy(e[ref_index-1])
                if lst_hyp[hyp_index-1] != lst_ref[ref_index-1]:
                    cur_e[ref_index].subs += 1
                cur_e[ref_index].total_cost = sub_err
            elif del_err < ins_err:
                cur_e[ref_index] = copy(cur_e[ref_index-1])
                cur_e[ref_index].total_cost = del_err
                cur_e[ref_index].dels += 1
            else:
                cur_e[ref_index] = copy(e[ref_index])
                cur_e[ref_index].total_cost = ins_err
                cur_e[ref_index].ins += 1

        for i in range(len(e)): e[i] = copy(cur_e[i])

    return e[-1].ins, e[-1].subs, e[-1].dels, e[-1].total_cost;

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_file', 
                    help='Path to predictions matrix')
    parser.add_argument('transcription_file', 
                    help='Path to actual transcripts file')

    arguments = parser.parse_args()

    f = open(arguments.prediction_file)
    prediction_lines = f.readlines()
    f.close()

    f = open(arguments.transcription_file)
    transcription_lines = f.readlines()
    f.close()

    assert(len(transcription_lines) == len(prediction_lines))

    word_errs, num_ref, ins, dels, subs = 0, 0, 0, 0, 0

    for ref_line, pred_line in zip(transcription_lines, prediction_lines):
        utt_id, ref = ref_line.rstrip("\n").split(" ", 1)
        ins_cur, subs_cur, dels_cur, tot_cur = LevenshteinEditDistance(ref, 
                                                    pred_line.rstrip("\n"))
        ins += ins_cur
        dels += dels_cur
        subs += subs_cur
        word_errs += tot_cur
        num_ref += len(ref.split(" "))

sys.stderr.write("Phone Error Rate (PER) %.2f %d/%d [INS, DELS, SUBS] = [%d, %d, %d]\n"\
         %(word_errs*100.0/num_ref, word_errs, num_ref, ins, dels, subs))
