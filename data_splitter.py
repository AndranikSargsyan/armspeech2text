import pandas as pd
from sklearn.utils import shuffle


if __name__ == '__main__':
    full_df = pd.read_csv(os.path.join(self.data_root, 'validated.tsv'), sep='\t')
    full_df = shuffle(full_df)
    full_df[:500].to_csv('my_train.tsv', sep='\t', index=False)
    full_df[-500:].to_csv('my_val.tsv', sep='\t', index=False)
