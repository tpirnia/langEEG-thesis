import sys,os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

# functions: (copied from: ch2_attempt6_tg_plots_UR.ipynb)
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def tg_plot(ax, 
            acc_array,
            subject, 
            params,
            cmap = 'Reds',
            title="",
            vmin=50,
            vmax=100,
            tw=np.arange(0,960,10),
            tick_pad=5,
            bins=5): #,extent=[0, 1000, 0, 1000]):
    '''
        vmin/max : range of colormap
        origin : Place the [0, 0] index of the array in the upper left or lower left
        corner of the axes. The convention 'upper' is typically used for
        matrices and images.
        extent : scalars (left, right, bottom, top), optional
        The bounding box in data coordinates that the image will fill.
    '''
    txmin = tw[len(tw)-acc_array.shape[0]]
    tymin = tw[len(tw)-acc_array.shape[1]]
    txmax = tw[-1]
    tymax = tw[-1]
    im = ax.imshow(acc_array, 
                   cmap=cmap,
                   vmin=vmin, 
                   vmax=vmax,
                   origin='lower',
                   extent=[txmin, txmax,tymin, tymax])
    if not title:  title = 'Subject '+ str(subject)
    
    ax.set(ylabel='train time (ms)', 
           ylim=[tymin-tick_pad,
                 tymax+tick_pad], #xlabel='test time (ms)',
           xlim=[txmin-tick_pad, 
                 txmax+tick_pad],
           title=title)
    ax.xaxis.set_ticks_position('bottom')
    plt.locator_params(nbins=bins)
    ax.set_aspect('equal', adjustable='box')
    cbar = add_colorbar(im, 
                        aspect=15, 
                        pad_fraction=0.5)
    cbar.set_label(rotation=270,
                   label='Accuracy (%)',
                   labelpad=18,
                   fontsize=18,
                  )
    #cbar.set_ticks(ticks=[.6, .8, 1])
    cbar.ax.tick_params(labelsize=16,pad=5)
    #ax.set_aspect('equal', adjustable='box')
    #print(cbar.ax.get_position().bounds)
    ax.tick_params(grid_linewidth=.75, grid_color='gray', grid_alpha=0.4)
    return ax

if __name__ == '__main__':
    subject = sys.argv[1]
    thresh = float(sys.argv[2])
    train_subject='-loo'
    print(subject)
    
    datapath='/project_data/volume0/tpirnia/thesis_repository/results/TG/'
    clusters_fname = datapath + 'clusters/sig_threshold_accuracy_{}/{}_trained{}.npy'.format(thresh,subject, train_subject)
    
    fig_fname = datapath + 'clusters/figures_threshold_accuracy_{}/{}_trained{}.png'.format(thresh,subject, train_subject)
    fill_path(fig_fname)
    
    sig_tg = np.load(clusters_fname)
    #for plotting:
    with mpl.rc_context(params):
        plt.style.use('seaborn-darkgrid')
        plt.style.use('seaborn-poster')
        fig, axs = plt.subplots(1,1, sharex=True, sharey=True, squeeze=False, figsize=(6,14))
        ax = axs.ravel()
        sig_tg = np.round(sig_tg * 100,decimals=1)
        fig_title=('Participant {} (p<0.05)'.format(int(subject[2:])))
        plt.gca()
        tg_plot(ax[0],
                sig_tg, 
                subject=subject,
                params=params,
                title=fig_title,
                bins=6)
        ax[0].set(xlabel='test time (ms)', ylabel='train time (ms)')
        ax[0].tick_params(axis='x', which='major', pad=2.5, labelsize=16)
        ax[0].tick_params(axis='y', which='major', pad=4, labelsize=16)
        plt.savefig(fig_fname,
                    dpi=450,
                    bbox_inches = "tight")
        plt.show()