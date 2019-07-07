import pandas as pd
from execute_ import tracking_params


def evaluate(dataset, to_latex_table=False, *args, **kwargs):
    df = pd.read_csv('pre_trained/' + dataset + 'log.csv', index_col=['run'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('\n\t\t\tNumber of runs\n')
        print(df.groupby(tracking_params).count()['training_epochs'])
        print('\n\t\t\tWith average results\n')
        print(df.groupby(tracking_params).mean())
        print(df.groupby(tracking_params).std())

    if to_latex_table:
        print(df.groupby(tracking_params).mean().to_latex(*args, **kwargs))
        print(df.groupby(tracking_params).std().to_latex(*args, **kwargs))


if __name__ == '__main__':
    evaluate('pubmed')
