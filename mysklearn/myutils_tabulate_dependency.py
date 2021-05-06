# tabulate utils are seperate to simplify heroku deployment

from tabulate import tabulate



def format_confusion_matrix(confusion_matrix, labels):
    '''Formats a confusion matrix

    Args:
        confusion_matrix (list of list): confusion matrix
        labels (list of str): values to predict
    '''
    columns = labels + ['Total', 'Recognition (%)']
    totals = [sum(confusion_matrix[i]) for i in range(len(confusion_matrix))]
    recognition = [(confusion_matrix[i][i] / totals[i]) * 100 if totals[i] != 0 else 0
                   for i in range(len(confusion_matrix))]
    confusion_matrix_format = [[str(labels[i]).ljust(3) + ' | '] + confusion_matrix[i] +
                               [totals[i], recognition[i]]
                               for i in range(len(labels))]
    return tabulate(confusion_matrix_format, headers = columns)
