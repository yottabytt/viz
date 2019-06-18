import matplotlib
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors

class Viz:

    def __init__(self, exps, n_epochs, x_label, y_label, title, fig_name):
        '''
        supports at max 6 exps because of currently 
        supported 6 constrasting base colors
        '''
        self.exps = exps
        self.n_epochs = n_epochs
        self.colors = [k for k in dict(mcolors.BASE_COLORS) if k != 'w' and k != 'k']
        self.line_styles = {
            'solid': '-',
            'dotted': ':',
            'solid_dotted': '-.'
        }
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.fig_name = fig_name

    def plot(self):
        fig, ax = plt.subplots()
        x = list(range(self.n_epochs))
        for idx, exp in enumerate(self.exps):
            for exp_set in exp['sets']:
                if exp_set == 'train':
                    ax.plot(x, exp['sets'][exp_set], color=self.colors[idx],
                                label="train [%s]" % (exp['label']), 
                                linestyle=self.line_styles['solid'])
                elif exp_set == 'valid':
                    ax.plot(x, exp['sets'][exp_set], color=self.colors[idx],
                                label="valid [%s]" % (exp['label']),
                                linestyle=self.line_styles['dotted'])
        
        ax.set(xlabel=self.x_label, ylabel=self.y_label)
        ax.set_title(self.title)
        ax.set_ylim([10, 80])
        ax.grid()
        ax.legend(loc='upper right', shadow=True)
        fig.savefig("figs/%s.png" % (self.fig_name))