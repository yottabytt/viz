import math
import statistics

from lm_log_parser import LMLogParser
from viz import Viz

class VizWrapper:

    @staticmethod
    def get_lmlp(file_loc, version='my_mos'):

        if version == 'my_mos':
            n_splits_valid = 6
            loss_o_pos_train = 6
            loss_o_pos_valid = 4
            val_i_pos_train = 1
            val_i_pos_valid = 2
        elif version == 'tcc_hsm':
            n_splits_valid = 8
            loss_o_pos_train = 4
            loss_o_pos_valid = 6
            val_i_pos_train = 3
            val_i_pos_valid = 2

        return LMLogParser(
                file_loc = file_loc,
                version = version,
                n_splits_train = 8,
                n_splits_valid = n_splits_valid,
                n_splits_test = 5,
                loss_o_pos_train = loss_o_pos_train,
                loss_o_pos_valid = loss_o_pos_valid,
                loss_o_pos_test = 3,
                val_i_pos_train = val_i_pos_train,
                val_i_pos_valid = val_i_pos_valid,
                val_i_pos_test = 1
            )

    @staticmethod
    def get_exp(label, lmlp, intvl_len, color):

        lmlp.parse()
        
        if lmlp.version == 'my_mos':
            intvl_train_loss = [float(x) for x in lmlp.parsed['intvl_train_loss']]
            epoch_train_loss = [statistics.mean(intvl_train_loss[x:x+intvl_len]) for x in range(0, len(intvl_train_loss),intvl_len)]
            epoch_train_ppl = [math.exp(x) for x in epoch_train_loss]
        elif lmlp.version == 'tcc_hsm':
            epoch_train_ppl = [float(x) for x in lmlp.parsed['epoch_train_ppl']]
        
        epoch_valid_ppl = [float(x) for x in lmlp.parsed['epoch_valid_ppl']]

        return {
            'label': label,
            'sets': {
                'train': epoch_train_ppl,
                'valid': epoch_valid_ppl
            },
            'color': color
        }

    @staticmethod
    def plot(exps, n_epochs, fig_name, y_lim):

        viz = Viz(
            exps = exps,
            n_epochs = n_epochs,
            x_label = 'epoch',
            y_label = 'perplexity',
            x_lim = None,
            y_lim = y_lim,
            title = '',
            fig_name = fig_name
        )
        viz.plot()
