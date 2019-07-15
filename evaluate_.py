import pandas as pd
from execute_ import tracking_params


def evaluate(dataset, to_latex_table=False, *args, **kwargs):
    '''
    This function opens the pretrained/{}log.csv files and calculates the average and standard deviation of
    each setup.
    :param dataset: String. Name of the dataset in lowercase
    :param to_latex_table: If True. Uses pandas to_latex function to generate the table.
    :param args: args to pass to the pandas to_latex function
    :param kwargs: kwargs to pass to the pandas to_latex function
    :return: Prints the results in the console.
    '''
    df = pd.read_csv('pre_trained/' + dataset + 'log.csv', index_col=['run'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('\n\t\t\tNumber of runs\n')
        print(df.groupby(tracking_params).count()['training_epochs'])
        print('\n\t\t\tWith average results\n')
        print(df.groupby(tracking_params).mean())
        print('\n\t\t\tWith std results\n')
        print(df.groupby(tracking_params).std())

    if to_latex_table:
        print(df.groupby(tracking_params).mean().reset_index(['dataset', 'lr', 'l2_coef', 'nonlinearity', 'param_attn_drop', 'param_ffd_drop'], drop=True).to_latex(*args, **kwargs))

        print(df.groupby(tracking_params).std().reset_index(['dataset', 'lr', 'l2_coef', 'nonlinearity', 'param_attn_drop', 'param_ffd_drop'], drop=True).to_latex(*args, **kwargs))


if __name__ == '__main__':
    evaluate('cora', to_latex_table=False, multirow=False, longtable=False, columns=['test_accuracy'], float_format='%.3f%%')
    evaluate('citeseer', to_latex_table=False, multirow=False, longtable=False, columns=['test_accuracy'], float_format='%.3f%%')
    evaluate('pubmed', to_latex_table=False, multirow=False, longtable=False, columns=['test_accuracy'], float_format='%.3f%%')
    evaluate('ppi', to_latex_table=False, multirow=False, longtable=False, columns=['test_accuracy'], float_format='%.3f%%')
