##############################################
# Programmer: Ciara Patterson
# Class: CPSC 322-02, Spring 2021
# Programming Assignment #3
# 2/25/2021
#
#
# General use functions for plotting data.
##############################################
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 
import matplotlib.pyplot as plt
import matplotlib as mpl

import mysklearn.myutils
import mysklearn.myutils as utils

# set resolution and style for matplotlib
mpl.rcParams.update({'font.size': 10}) 
mpl.rcParams['figure.dpi']= 150
mpl.rc("savefig", dpi=150)

def plot_frequency_diagram(table, group_by_col, rotate_x = True):
    '''Plots a frequency diagram for each of the unique values in the group by column in the table 
    Args: 
        table (MyPyTable): table with group_by_column 
        group_by_col (str): column name in table 
    '''
    category, frequencies = table.group_by_count(group_by_col)
    plt.figure(figsize = (10,5))
    plt.rc('font', size = 10) 
    plt.style.use('seaborn-whitegrid')
    plt.bar([str(x) for x in category], frequencies)
    plt.title('Frequency Count for ' + group_by_col)
    if rotate_x:
        plt.xticks(rotation = 90, horizontalalignment = 'center')
    plt.show()

def plot_bar_chart(labels, y, rotate_x = True):
    plt.figure()
    plt.bar(labels, y)
    if rotate_x:
        plt.xticks(rotation = 90, horizontalalignment = 'center')
    plt.show()
    
def plot_pie_chart(x, y):
    plt.figure()
    plt.pie(y, labels = x, autopct = "%1.1f%%")
    plt.show()
    

def plot_histogram(data, data_name, bins = 10, rotate_x = False):
    plt.figure()
    plt.hist(data, bins = bins)
    plt.xlabel(data_name)
    if rotate_x:
        plt.xticks(rotation = 90, horizontalalignment = 'center')
    plt.show()
    

def plot_overlapping_histogram(data, labels, bins = 10, rotate_x = False):
    '''Plot overlapping histogram 
    
    Args:
        data (list of lists): lists of data to plot on a histogram
        bins (int): number of bins in the histogram
    '''
    plt.figure()
    for i in range(len(data)):
        plt.hist(data[i], bins = bins, alpha = 0.3, label = labels[i])
    if rotate_x:
        plt.xticks(rotation = 90, horizontalalignment = 'center')
    plt.legend(loc = 'upper left', fontsize = 8)
    plt.show()

def plot_scatterplot(x, y, x_name, y_name, linear_reg = True, legend_loc = 'upper right'):
    plt.figure()
    plt.scatter(x, y)
    if linear_reg:
        # plot the linear regression line on the plot
        m, b = utils.calculate_slope(x, y)
        plt.plot([min(x), max(x)], [m*min(x) + b, m * max(x) + b], c = 'r', lw = 5)
        # add the correlation coefficient and covariance to the graph 
        r = utils.calculate_correlation(x, y)
        cov = utils.calculate_covariance(x, y)
        if legend_loc == 'upper right':
            plt.annotate("$r = {}$ \n cov = {}".format(round(r, 2), round(cov, 2)), xy = (.95, .90), 
                         xycoords = "axes fraction", horizontalalignment = "right", color = "blue")
        if legend_loc == 'upper left':
            plt.annotate("$r = {}$ \n cov = {}".format(round(r, 2), round(cov, 2)), xy = (.20, .90), 
                         xycoords = "axes fraction", horizontalalignment = "right", color = "blue")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

def plot_boxplot(data):
    plt.figure()
    plt.boxplot(data)
    plt.show()
    
def plot_dictionary_boxplot(data_dict, boxplot_title, rotate_x = True):
    plt.figure()
    plt.boxplot(data_dict.values())
    if rotate_x:
        plt.xticks([i+1 for i in range(len(data_dict.keys()))], data_dict.keys(), 
                   rotation = 90, horizontalalignment = 'center')
    else:
        plt.xticks([i+1 for i in range(len(data_dict.keys()))], data_dict.keys())
    plt.title(boxplot_title)
    plt.show()
