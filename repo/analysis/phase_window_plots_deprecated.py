from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
from itertools import product

@dataclass
class Config:
    subjects: List[str]
    project_path: Path
    times: np.ndarray
    mask_distance: int
    plot_params: Dict
    
    @classmethod
    def create_default(cls) -> 'Config':
        subjects = "EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split()
        # subjects = ["EP07"]
        # project_path = Path('/project_data/volume0/tpirnia/thesis_repository')
        project_path = Path('/Users/tpirnia/Library/CloudStorage/Dropbox/Repository/LanguageProduction/COPY_FROM_SERVER') # need to change
        times = np.arange(0, 960, 10)
        mask_distance = 25
        
        plot_params = {
            'lines.markersize': 6,
            'axes.labelsize': 20,
            'axes.titlesize': 20,
            'axes.labelpad': 8,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'xtick.major.pad': 7,
            'ytick.major.pad': 6,
            'legend.labelspacing': .5,
            'legend.handletextpad': .175,
            'legend.title_fontsize': 20,
            'legend.fontsize': 20,
            'figure.figsize': (8, 10),
            'figure.subplot.hspace': .95,
            'figure.subplot.wspace': .95,
            'image.origin': 'lower',
            'figure.autolayout': False,
            'scatter.marker': 'o',
            'savefig.dpi': 450,
            'savefig.bbox': 'tight',
            'savefig.transparent': True,
        }
        
        return cls(subjects, project_path, times, mask_distance, plot_params)

class TemporalGeneralization:
    def __init__(self, config: Config):
        self.config = config
        self.mask_matrix = self._create_mask_matrix()
    
    def _create_mask_matrix(self) -> np.ndarray:
        """Create mask matrix for temporal generalization."""
        n_times = len(self.config.times)
        mask = np.zeros((n_times, n_times))
        for i, j in product(range(n_times), range(n_times)):
            if abs(i - j) > self.config.mask_distance:
                mask[i, j] = 1
        return mask
    
    def load_data(self, test_subject: str, trained_by: str) -> np.ndarray:
        """Load temporal generalization data."""
        fname = self.config.project_path / 'results' / 'TG_phase' / f'{test_subject}_trained-{trained_by}.npy'
        return np.load(fname)
    
    def apply_mask(self, data: np.ndarray) -> np.ndarray:
        """Apply mask to temporal generalization matrix."""
        masked_data = deepcopy(data)
        masked_data[self.mask_matrix == 1] = 0
        return masked_data
    
    def threshold(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Apply threshold, set to np.nan where < thresholdto temporal generalization matrix."""
        data[ np.where(data < threshold) ] = np.nan
        return data
    
    def find_max_along_test_time(self, data: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find maximum values along test time dimension."""
        inds = np.where(data.max(axis=0) > threshold)[0]
        argmax_tr = data.argmax(axis=0)
        max_tr = data.max(axis=0)
        thresh_test_times = self.config.times[inds]
        thresh_train_times = self.config.times[argmax_tr[inds]]
        thresh_acc = max_tr[inds]
        return thresh_acc, thresh_train_times, thresh_test_times
    
    def get_peaks(self, thresh_acc: np.ndarray, thresh_train_time: np.ndarray, 
                 thresh_test_times: np.ndarray) -> Tuple[float, float, float]:
        """Get peak accuracy and corresponding times."""
        ind_peak = np.argmax(thresh_acc)
        return (thresh_acc[ind_peak], 
                thresh_train_time[ind_peak], 
                thresh_test_times[ind_peak])

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.tg = TemporalGeneralization(config)
    
    def process_subject(self, subject: str) -> Tuple[List, List]:
        """Process data for a single subject."""
        within_results = []
        loo_results = []
        
        # Load data
        within_data = self.tg.load_data(subject, 'subject')
        loo_data = self.tg.load_data(subject, 'loo')
        
        # Process within-subject data
        within_masked = self.tg.apply_mask(within_data)
        within_thresh = (
            self.tg.find_max_along_test_time(within_masked, 0.55),
            self.tg.find_max_along_test_time(within_masked, 0.60)
        )
        
        if len(within_thresh[0][0]) > 0:
            peaks = self.tg.get_peaks(*within_thresh[0])
            within_results.append([
                subject, subject, 
                len(within_thresh[0][0]), len(within_thresh[1][0]),
                *peaks
            ])
        
        # Process LOO data
        loo_masked = self.tg.apply_mask(loo_data)
        loo_thresh = (
            self.tg.find_max_along_test_time(loo_masked, 0.55),
            self.tg.find_max_along_test_time(loo_masked, 0.60)
        )
        
        if len(loo_thresh[0][0]) > 0:
            peaks = self.tg.get_peaks(*loo_thresh[0])
            loo_results.append([
                subject, 'LOO',
                len(loo_thresh[0][0]), len(loo_thresh[1][0]),
                *peaks
            ])
        
        return within_results, loo_results

class Visualizer:
    def __init__(self, config: Config):
        self.config = config
    
    def create_dataframe(self, within_data: List, loo_data: List) -> pd.DataFrame:
        """Create and format DataFrame from results."""
        cols = ['test_subject', 'train_subject', 'n_thresh55', 'n_thresh60',
                'acc_train_peak', 't_train_peak', 't_test_peak']
        
        within_df = pd.DataFrame(within_data, columns=cols)
        loo_df = pd.DataFrame(loo_data, columns=cols)
        
        within_df['training_data'] = 'within participant'
        loo_df['training_data'] = 'across participants (LOO)'
        
        df = pd.concat([loo_df, within_df])
        
        # Format categorical columns
        cat_type = pd.CategoricalDtype(categories=self.config.subjects + ['LOO'], ordered=True)
        df['train_subject'] = df['train_subject'].astype(cat_type)
        
        cat_type = pd.CategoricalDtype(categories=self.config.subjects, ordered=True)
        df['test_subject'] = df['test_subject'].astype(cat_type)
        
        # training_data = set(df['training_data'][::-1])
        cat_type = pd.CategoricalDtype(categories=['within participant', 'across participants (LOO)'], ordered=True)
        df['training_data'] = df['training_data'].astype(cat_type)
        
        return df
    
    def create_plot(self, df: pd.DataFrame, save_path: Path) -> None:
        """Create and save visualization."""
        with mpl.rc_context(self.config.plot_params):
            plt.style.use(['seaborn-v0_8-darkgrid', 'seaborn-v0_8-poster'])
            fig, ax = plt.subplots(figsize=(8.7, 5.27))
            plt.ylim(0, 1)
            
            sns.barplot(
                y='acc_train_peak',
                x='training_data',
                data=df,
                ax=ax,
                errwidth=3,
                palette={
                    'within participant': '#FF9933',
                    'across participants (LOO)': '#FF4D4D'
                },
                errorbar=('ci', 95)
            )
            
            locs, _ = plt.yticks()
            ax.xaxis.labelpad = 17
            ax.set(
                ylabel='peak test accuracy (%)',
                yticks=locs,
                yticklabels=list(map(int, locs * 100)),
                xlabel=''
            )
            
            plt.savefig(save_path, dpi=450, bbox_inches="tight")
            plt.close()

    def create_histogram_plot(self, df: pd.DataFrame, save_path: Path) -> None:
            """Create and save histogram visualization of test time distributions."""
            subtitles = ['within participant classification', 
                        'across participants - LOO classification']
            cbar = ['#FF9933', '#FF4D4D']
            with mpl.rc_context(self.config.plot_params):
                plt.style.use(['seaborn-v0_8-darkgrid', 'seaborn-v0_8-poster'])

                g = sns.FacetGrid(
                    df, 
                    col="training_data",
                    height=5,
                    aspect=1.5, 
                    hue="training_data",
                    palette={
                        'within participant': '#FF9933',
                        'across participants (LOO)': '#FF4D4D'
                    }
                )

                g.map(
                    sns.histplot,
                    't_test_peak',
                    stat='percent',
                    binwidth=200,
                    binrange=(0, 1000),
                    edgecolor="white",
                    linewidth=1
                )

                # Format each facet
                for i, ax in enumerate([g.facet_axis(0,0), g.facet_axis(0,1)]):
                    # Set title and limits
                    ax.set_title(subtitles[i])
                    ax.set_ylim(0, 50)

                    # Add percentage labels on bars
                    for patch in ax.patches:
                        height = patch.get_height()

                        # Determine text position and color
                        if height == 0:
                            vert_offset = 0.5
                            text_color = 'gray'
                        else:
                            vert_offset = -3.5
                            text_color = 'white'

                        # Add text
                        ax.text(
                            x=patch.get_x() + (patch.get_width()/2),
                            y=height + vert_offset,
                            s=f'{height:.0f}%',
                            ha='center',
                            fontsize=20,
                            color=text_color,
                            fontweight="bold"
                        )

                    # Set labels
                    ax.set_xlabel('test time (ms)', fontsize=20)
                    ax.set_ylabel('% of total subjects', fontsize=20)

                    # Adjust tick parameters
                    ax.tick_params(axis='both', pad=6)

                plt.savefig(save_path, dpi=450, bbox_inches="tight")
                plt.close()
def main():
    # Initialize configuration
    # Initialize configuration
    config = Config.create_default()
    
    # Initialize processors
    processor = DataProcessor(config)
    visualizer = Visualizer(config)
    
    # Process data
    within_results = []
    loo_results = []
    
    for subject in config.subjects:
        within_data, loo_data = processor.process_subject(subject)
        within_results.extend(within_data)
        loo_results.extend(loo_data)
    
    # Create DataFrame
    df = visualizer.create_dataframe(within_results, loo_results)
    
    # Create visualizations
    visualizer.create_plot(
        df, 
        config.project_path / 'results' / 'figures' / '_Phase-3-Comparison_of_Peak_Acuracy_Within_vs_Across_Participants.png'
    )
    
    visualizer.create_histogram_plot(
        df,
        config.project_path / 'results' / 'figures' / '_Phase-4-Distribution_of_Peak_Test_Time_Within_vs_Across_Participants.png'
    )
    # Save data
    df.to_csv(config.project_path / 'results' / 'by_analysis_number'/ 'analysis-4-peak_window_accuracy.csv', index=False)

if __name__ == "__main__":
    main()