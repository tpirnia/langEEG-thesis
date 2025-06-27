import sys,os
import numpy as np
import copy 

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits import axes_grid1
from matplotlib import colors

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname), exist_ok=True)

#plot parameters:
params = {'lines.markersize':6,
          'axes.labelsize': 20,
          'axes.titlesize':20,
          'axes.labelpad':8,
          'xtick.labelsize' :16,
          'ytick.labelsize' :16,
          'xtick.major.pad':7, 
          'ytick.major.pad':6, 
          'legend.labelspacing':.5,
          'legend.handletextpad':.175,
          'legend.title_fontsize':20,
          'legend.fontsize':20,
          'figure.figsize':(8,10),
          'figure.subplot.hspace': .95,
          'figure.subplot.wspace': .95,
          'image.origin': 'lower',
          'figure.autolayout': False,
          'scatter.marker':'o',
          'savefig.dpi': 450,
          'savefig.bbox': 'tight',
          'savefig.transparent': True,
         }


if __name__ == '__main__':
    subject = sys.argv[1]
    
    thresh = np.float(sys.argv[2])
    datapath='/project_data/volume0/tpirnia/thesis_repository/results/TG/'
    train_subject='-subject'
    # if (len(sys.argv)>3) and (sys.argv[3]=='loo'):
    #     train_subject = '-' + sys.argv[3]
    #     datapath= datapath[:-1] + '_interp_loo/'
    # else: 
    #     train_subject = subject
        
    print("{} | {}".format(subject,train_subject))
    

    out_fname = (datapath 
                     + 'clusters/' 
                     + '{}_threshold_accuracy_{}/{}_trained{}{}'.format('null',
                                                                        thresh,
                                                                        subject, 
                                                                        train_subject, 
                                                                        '-null'))

    fig_fname = (datapath 
                 + 'clusters/' 
                 + 'figures_threshold_accuracy_{}/null_cluster_distribution/{}_trained{}{}.png'.format(thresh,
                                                                                                      subject, 
                                                                                                      train_subject,
                                                                                                      '-null'))
    fill_path(fig_fname)
    max_clusts = np.load(out_fname +'-maxdist.npy') 

    with mpl.rc_context(params):
        plt.style.use('seaborn-darkgrid')
        plt.style.use('seaborn-poster')

        plt.figure(figsize=(7,5))
        plt.title('Distribution of maximum null clusters:\nSubject {}'.format(int(subject[2:])))
        plt.hist(max_clusts,bins=20)
        plt.xlabel('cluster size')
        plt.ylabel('count')
        plt.savefig(fig_fname,
            dpi=450,
            bbox_inches = "tight")
        plt.show()