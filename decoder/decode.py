# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

# No need to modify file.
import argparse, time, os, sys, scipy.io
from numpy import log, zeros, genfromtxt, exp

def path_leaf(path):
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)

def read_phones(phone_file):
    f = open(phone_file, 'r')
    lines = f.readlines()
    f.close()
    phones = [phone.rstrip('\n').replace(' ', '') for phone in lines]
    phone_map = {}
    for index, phone in enumerate(phones):
        phone_map[phone] = index
    return phones, phone_map

def read_transition_counts(trans_file):
    trans_probs = genfromtxt(trans_file)
    return trans_probs


def forward(log_probs, log_trans_all, lang_wt):
    num_phones = log_trans_all.shape[0] - 2
    log_trans = log_trans_all[:num_phones, :num_phones]

    state_time_log_probs = zeros((num_phones, log_probs.shape[1]))
    state_time_log_probs[:,0] = lang_wt*log_trans_all[num_phones,:num_phones] + \
                                                          log_probs[:,0]
    num_frames = log_probs.shape[1]
    for frame_num in range(1, num_frames):
        log_state_joint = (lang_wt * log_trans + \
                       state_time_log_probs[:,frame_num-1].reshape(-1,1)).max(axis=0)
        state_time_log_probs[:,frame_num] = log_state_joint + log_probs[:,frame_num]

    log_state_joint = log_trans_all[:num_phones,-1] + state_time_log_probs[:,-1]
    max_final_state = log_state_joint.argmax()

    return state_time_log_probs, max_final_state


def backwards(state_time_log_probs, log_trans_all, max_final_state, lang_wt):
    num_phones = log_trans_all.shape[0] - 2
    log_trans = log_trans_all[:num_phones, :num_phones]

    num_frames = state_time_log_probs.shape[1]
    reverse_path = [max_final_state]

    for frame_num in range(num_frames-2, -1, -1):
        state_log_probs = state_time_log_probs[:,frame_num-1]
        log_trans_to_state = log_trans[:, max_final_state]
        max_final_state = (lang_wt * log_trans_to_state + state_log_probs).argmax()
        reverse_path.append(max_final_state)

    reverse_path.reverse()
    return reverse_path

def decode(log_probs, trans_counts, lang_wt):
    small = 1
    trans_counts_disc = trans_counts + small * (1 + trans_counts.sum(axis=1).reshape(-1,1))
    trans_probs = trans_counts_disc / trans_counts_disc.sum(axis=1).reshape(-1,1)
    log_trans_all = log(trans_probs)

    state_time_log_probs, max_final_state = forward(log_probs, log_trans_all, lang_wt)
    state_path = backwards(state_time_log_probs, log_trans_all, max_final_state, lang_wt)
    return state_path

def cleanup(state_path):
    last_state = state_path[0]
    clean_path = [last_state]
    for state in state_path:
        if state != last_state:
            clean_path.append(state)
            last_state = state

    return clean_path

parser = argparse.ArgumentParser()
parser.add_argument('--lang_wt', type=float, default = 10.0, 
                     help='language model weight')
parser.add_argument('phone_file', help='Path to phones file')
parser.add_argument('trans_file', help='Path to transition probabilities')
parser.add_argument('prediction_file', help='Path to predictions matrix')
parser.add_argument('output_file', help='Output file')

arguments = parser.parse_args()

phones, phone_map = read_phones(arguments.phone_file)
trans_counts = read_transition_counts(arguments.trans_file)

predictions_dict = scipy.io.loadmat(arguments.prediction_file)
predictions_mat = predictions_dict['predictions']
utt_indices = predictions_dict['utt_indices']

num_utterances = utt_indices.shape[0]
f = open(arguments.output_file, 'w')
for utt_num in range(num_utterances):
    s, e = utt_indices[utt_num,:]
    predictions = predictions_mat[:, s:e]
    state_path = cleanup(decode(predictions, trans_counts, 
                                arguments.lang_wt))

    f.write(" ".join([phones[x] for x in state_path]))
    f.write("\n")
