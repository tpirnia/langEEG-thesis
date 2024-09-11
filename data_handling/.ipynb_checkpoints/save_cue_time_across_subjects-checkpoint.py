import numpy as np

f_in = '/project_data/volume0/tpirnia/data/EEGWS/data/EP03/data_c_avg.npz'
f_out = '/project_data/volume0/tpirnia/data/EEGWS/data/across_subjects/cue_time.npy'
cue_time = np.load(f_in)['t'][0]
np.save(f_out,cue_time)