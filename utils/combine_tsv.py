import pandas as pd

dev = pd.read_csv('./korsts/sts-dev.tsv', delimiter='\t', on_bad_lines='skip')
test = pd.read_csv('./korsts/sts-test.tsv', delimiter='\t', on_bad_lines='skip')
train = pd.read_csv('./korsts/sts-train.tsv', delimiter='\t', on_bad_lines='skip')

columns_to_keep = ['score', 'sentence1', 'sentence2']
dev = dev[columns_to_keep]
test = test[columns_to_keep]
train = train[columns_to_keep]

combined_df = pd.concat([dev, test, train])

combined_df.to_csv('./korsts/sts-total.tsv', sep='\t', index=False)