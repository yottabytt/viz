import re

class LMLogParser:
    
    def __init__(self, file_loc, n_splits_train,
                    n_splits_valid, n_splits_test, loss_o_pos_train,
                    loss_o_pos_valid, loss_o_pos_test, val_i_pos_train,
                    val_i_pos_valid, val_i_pos_test, version='my_mos'):

        self.file_loc = file_loc
        self.version = version
        self.n_splits_train = n_splits_train
        self.n_splits_valid = n_splits_valid
        self.n_splits_test = n_splits_test
        self.loss_o_pos_train = loss_o_pos_train
        self.loss_o_pos_test = loss_o_pos_test
        self.loss_o_pos_valid = loss_o_pos_valid
        self.val_i_pos_train = val_i_pos_train
        self.val_i_pos_valid = val_i_pos_valid
        self.val_i_pos_test = val_i_pos_test
        self.parsed = {
            'intvl_train_loss': [],
            'intvl_train_ppl': [],
            'epoch_valid_loss': [],
            'epoch_valid_ppl': [],
            'epoch_train_loss': [],
            'epoch_train_ppl': []
        }
        
    def parse_train_log(self, splits):
        self.parsed['intvl_train_loss'].append(
            splits[self.loss_o_pos_train].split()[self.val_i_pos_train].strip()
        )
        self.parsed['intvl_train_ppl'].append(
            splits[self.loss_o_pos_train + 1].split()[self.val_i_pos_train].strip()
        )


    def parse_valid_log(self, splits, version):
        self.parsed['epoch_valid_loss'].append(
            splits[self.loss_o_pos_valid].strip().split()[self.val_i_pos_valid]
        )
        self.parsed['epoch_valid_ppl'].append(
            splits[self.loss_o_pos_valid + 1].strip().split()[self.val_i_pos_valid]
        )

        if version == 'tcc_hsm':
            self.parsed['epoch_train_loss'].append(
                splits[self.loss_o_pos_train].strip().split()[self.val_i_pos_train]
            )
            self.parsed['epoch_train_ppl'].append(
                splits[self.loss_o_pos_train + 1].strip().split()[self.val_i_pos_train]
            )

    def parse(self):
        with open(self.file_loc, 'r') as f:
            for l in f:
                splits = re.split(r'\|', l)
                n_splits = len(splits)
                if self.version == 'tcc_hsm':
                    if(n_splits == self.n_splits_valid and
                        'end' in splits[2]):
                        self.parse_valid_log(splits, self.version)
                else:
                    if(n_splits == self.n_splits_train):
                        self.parse_train_log(splits)    
                    elif(n_splits == self.n_splits_valid):
                        self.parse_valid_log(splits, self.version)

if __name__ == '__main__':
    
    lmlp = LMLogParser(
        file_loc = '../results/test_exp/abo/log.txt',
        n_splits_train = 8,
        n_splits_valid = 6,
        n_splits_test = 5,
        loss_o_pos_train = 6,
        loss_o_pos_valid = 4,
        loss_o_pos_test = 3,
        val_i_pos_train = 1,
        val_i_pos_valid = 2,
        val_i_pos_test = 1
    )
    lmlp.parse()
    assert len(lmlp.parsed['epoch_valid_loss']) == 1000
    assert len(lmlp.parsed['epoch_valid_ppl']) == 1000
    assert len(lmlp.parsed['intvl_train_loss']) == 5000
    assert len(lmlp.parsed['intvl_train_ppl']) == 5000