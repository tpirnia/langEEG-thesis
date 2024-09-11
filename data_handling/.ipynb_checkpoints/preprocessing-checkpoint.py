import numpy as np

def trim_EEG(eeg, t, t_start=0, t_end=None):
    """ segments EEG and time [trials, channels, t_start:t_end]. t is an array of timestamps corresponding to 3rd dim of EEG data.
        """
    cue_index = np.flatnonzero(t[0]>=t_start)[0]
    if isinstance(t_end,type(None)):
        res_index = len(t[0])-1
    else:
        res_index = np.flatnonzero(t[0]<t_end)[-1]
    eeg = eeg[:,:,cue_index:res_index]
    t = t[:,cue_index:res_index]
    return eeg, t

def list_bad_channels(ch):
    bad_ch = np.unique([np.where(x==-1)[0] for x in ch])
    return bad_ch

def remove_bad_channels(ch, eeg):
    bad_ch = list_bad_channels(ch)
    return np.delete(eeg, bad_ch, axis=1)

def get_bad_channels(project_path, subject):
    ch = np.load(project_path+'data/'+subject+'/data_c_avg.npz')['ch']
    bad_ch = np.unique([np.where(x==-1)[0] for x in ch])
    return bad_ch