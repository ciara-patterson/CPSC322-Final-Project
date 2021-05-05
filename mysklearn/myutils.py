##############################################
# Programmer: Ciara Patterson
# Class: CPSC 322-02, Spring 2021
# Programming Assignment #4
# 4/28/2021
#
# General utility functions for PA7
##############################################

import math
from tabulate import tabulate
from collections import Counter
import itertools
import random


def calculate_mean(data):
    '''Calculates the mean of a list of data

    Args:
        data (list of floats): data whose mean you want
    Returns:
        (float): the mean of the dataset
    '''
    return sum(data) / len(data)

def calculate_slope(x, y):
    '''Calculates the slope for a linear regression line between the x and the y values

    Args:
        x (list of floats): independent variables
        y (list of floats): dependent variables corresponding to the x values

    Returns:
        m (float): the slope of the regression line
        b (float): the y-intercept of the regression line
    '''
    x_mean = calculate_mean(x)
    y_mean = calculate_mean(y)
    m = sum([(x[i] - x_mean)*(y[i] - y_mean) for i in range(len(x))]) / sum([(x[i] - x_mean)**2 for i in range(len(x))])
    b = y_mean - m * x_mean
    return m, b

def flatten_list(nested_list):
    '''Takes a 2D list and return a 1D list with the values in the nested list

    Args:
        nested_list (list of lists): list of lists whose items you wants

    Returns:
        combined_list (list): list of values in the inner lists of nested_list
    '''
    combined_list = []
    for inner_list in nested_list:
        combined_list.extend(inner_list)
    return combined_list

def categorical_distance(value1, value2):
    if value1 == value2:
        return 0
    else:
        return 1

def compute_euclidean_distance(v1, v2):
    '''Takes 2 vectors and computes the Euclidean distance between them

    Args:
        v1 (list of numeric values): the first vector
        v2 (list of numeric values): the second vector

    Returns:
        dist (float): Euclidean distance between the 2 vectors
    '''
    assert len(v1) == len(v2) # vectors must be in the same dimension
    dist = math.sqrt(sum([(v1[i] - v2[i]) ** 2 if isinstance(v1[i], (int, float))
                          else categorical_distance(v1[i], v2[i]) for i in range(len(v1))]))
    return dist

def map_val_to_cutoffs(val, cutoffs, bin_categories):
    '''Given a list of cutoffs and categories that respond to each bin, returns the category
    that the value corresponds to

    Args:
        val (float or int): the value whose category you want
        cutoffs (list of floats or ints): the cutoffs for each bin
        bin_categories (list): categories that respond to the bins
        Must be the same length as the cutoff lists

    Returns:
        bin_categories[i] (any value): the category that the value belongs to

    '''
    for i in range(0, len(cutoffs) - 1):
        if (i + 1) == (len(cutoffs) - 1):
            if val >= cutoffs[i] and val <= cutoffs[i+1]:
                return bin_categories[i]
        else:
            if val >= cutoffs[i] and val < cutoffs[i + 1]:
                return bin_categories[i]

def compute_accuracy(y_pred, y_test, error_rate = False):
    '''Compute the accuracy given a set of predicted values and the actual values.

    Args:
        y_pred (list of values): The list of the model's predictions
        y_test (list of values): The actual values
        error_rate (bool): whether to return the accuracy of the error

    Returns:
        (float): the percent accuracy of the error

    '''
    num_right = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            num_right += 1
    if error_rate:
        return (len(y_pred) - num_right) / len(y_pred)
    else:
        return num_right / len(y_pred)

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


class MinMaxScale:
    '''Used to scale data using a max and min normalization techniques.

    Attributes:
        orig_instances: data that is used to normalize the data
        attrs: each attribute represented in the inputted data
        mins: minimum values for each attribute, parallel to attrs
        maxs: maximum values for each attribute, parallel to attrs
    '''

    def __init__(self, orig_instances = None, attrs = None):
        self.orig_instances = orig_instances
        self.attrs = attrs
        # list of mins and maxs associated with each attribute
        self.mins = []
        self.maxs = []

    def get_mins_maxs(self):
        '''Stores the mins and maxs for each attribute'''
        # get the maxs and mins for each attribute
        for i in range(len(self.attrs)):
            values = [x[i] for x in self.orig_instances]
            min_val = min(values)
            max_val = max(values)

            # save the values in a list that is parallel to attrs
            self.mins.append(min_val)
            self.maxs.append(max_val)

    def normalize(self, new_instances):
        '''Normalizes any inputted data that resembles the original instances
            using the min and max values for each attribute
        '''
        assert len(new_instances[0]) == len(self.orig_instances[0])
        # now scale each attribute according the min and max values
        for i in range(len(self.attrs)):
            for j in range(len(new_instances)):
                normalized_val = (new_instances[j][i] - self.mins[i]) / (self.maxs[i] - self.mins[i])
                new_instances[j][i] = normalized_val

        return new_instances

    def revert(self, number, attr):
        '''Undoes the normalization of any number given its attribute'''
        attr_idx = self.attrs.index(attr)
        return (number * (self.maxs[attr_idx] - self.mins[attr_idx])) + self.mins[attr_idx]


def compute_attribute_domains(X_train, header):
    '''Computes the attribute domains given a list of training data

    Args:
        X_train (list of list): training data
        header (list of str): attribute names
    '''
    attribute_domains = {}
    for i in range(len(header)):
        all_values = [x[i] for x in X_train]
        unique_values = list(set(all_values))
        attribute_domains[header[i]] = unique_values
    return attribute_domains

def all_same_class(instances):
    '''Checks if every instances is the same class

    Args:
        instances (list of list): data where class label is the last item in the list
    '''
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False
    return True # if we get here, all instance labels matched the first label


def flatten_list(nested_list):
    '''Takes a 2D list and return a 1D list with the values in the nested list

    Args:
        nested_list (list of lists): list of lists whose items you wants

    Returns:
        combined_list (list): list of values in the inner lists of nested_list
    '''
    combined_list = []
    for inner_list in nested_list:
        combined_list.extend(inner_list)
    return combined_list

def group_by_attribute_values(instances, attribute, attribute_domains, header):
    '''Groups data according to the attribute values '''
    attr_index = header.index(attribute)
    attr_vals = {} # each key is a unique attribute value
    for value in attribute_domains[attribute]:
        attr_vals[value] = []

    for row in instances:
        value = row[attr_index]
        attr_vals[value].append(row)
    return attr_vals

def calculate_entropy(instances):
    '''Calculates entropy for a list of instances'''
    unique_classes = list(set([x[-1] for x in instances]))
    val_entropy = 0
    for class_val in unique_classes:
        num_class = len([x[-1] for x in instances if x[-1] == class_val])
        ratio = num_class / len(instances)
        val_entropy += -ratio * math.log2(ratio)
    return val_entropy

def select_attribute(instances, available_attributes, attribute_domains, header):
    '''Select the attribute to split on according to entropy'''
    entropies = []
    for attribute in available_attributes: # need to compare entropy for each attribute
        attr_instances = group_by_attribute_values(instances, attribute, attribute_domains, header)
        entropy = 0
        for val in attr_instances.keys():
            num_instances = len(instances)
            entropy += (len(attr_instances[val]) / num_instances) * calculate_entropy(attr_instances[val])
        entropies.append(entropy)
    sorted_entropies, sorted_attrs = (list(tup) for tup in zip(*sorted(zip(entropies, available_attributes))))
    return sorted_attrs[0]

def partition_instances(instances, split_attribute, attribute_domains, header):
    '''Paritions a list of instances according to a the value of the split_attribute'''
    # this is a group by split_attribute's domain, not by
    # the values of this attribute in instances
    # example: if split_attribute is "level"
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute) # 0
    # lets build a dictionary
    partitions = {} # key (attribute value): value (list of instances with this attribute value)
    # task: try this!
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions

# def all_same_class(instances):
#     '''Checks if all the instances belong to the same class

#     Returns:
#         True if instances are all the same class
#         False if not
#     '''
#     # assumption: instances is not empty and class label is at index -1
#     first_label = instances[0][-1]
#     for instance in instances:
#         if instance[-1] != first_label:
#             return False
#     return True # if we get here, all instance labels matched the first label

def create_leaf_node(partition, partitions, case):
    '''Creates either a regular leaf node (case 1) or a majority leaf node (case 2)

    Args:
        partition (list of lists): instances captured by this leaf node
        partitions (dict attribute_value:partition pairs): all instances captured by the level
        case (int): indicates whether this should be a majority vote leaf node or not
    Returns:
        leaf (list): list representation of the leaf node
    '''
    if case == 1: # all partitions are the same class
        class_value = partition[0][-1] # last item in the instance is the class
        num_instances = len(partition) # number of instances with that class
        total_instances =  len(flatten_list(partitions.values())) # total number of instances
        leaf = ["Leaf", class_value, num_instances, total_instances] # create node
    if case == 2: # partition is not all in the same class, use a majority vote
        # get most common class
        all_class_values = [x[-1] for x in partition]
        most_common_class = Counter(all_class_values).most_common(1)[0][0]
        num_instances = len(partition) # number of instances
        total_instances =  len(flatten_list(partitions.values())) # total number of instances
        leaf = ["Leaf", most_common_class, num_instances, total_instances] # create node
    return leaf

def prepend_attribute_label(table, header):
    '''Prepends attribute labels to each value in the instance

    Args:
        table (list of list): table of instances
        header (list of str): attribute names
    '''
    for row in table:
        for i in range(len(row)):
            row[i] = header[i] + " = " + str(row[i])

def check_row_match(terms, row):
    '''Checks if the terms are in the row

    Args:
        terms (list): terms to check for
        row (list): row in table

    Returns:
        1 if all the terms in terms are in row, 0 otherwise
    '''
    for term in terms:
        if term not in row:
            return 0
    return 1

def compute_rule_counts(rule, table):
    '''Computes the counts of each rule

    Args:
        rule (dict): representation of the rule
        table (list of list): table being mined

    Returns:
        Nleft (int): count of the left hand side set in the table
        Nright (int): count of the right hand side set in the table
        Nboth (int): count of the left and right hand side set in the table
        Ntotal (int): count of rows in the table
    '''
    Nleft = Nright = Nboth = 0 # accumulators
    Ntotal = len(table)
    for row in table:
        # Nleft: is lhs a subset of the row?
        Nleft += check_row_match(rule["lhs"], row)
        Nright += check_row_match(rule["rhs"], row)
        Nboth += check_row_match(rule["lhs"] + rule["rhs"], row)

    return Nleft, Nright, Nboth, Ntotal

def compute_rule_interestingness(rule, table):
    '''Computes measures of rule interestingness, including support, confidence, and lift

    Args:
        rule (dict): representation of the rule
        table (list of list): table being mined
    '''
    Nleft, Nright, Nboth, Ntotal = compute_rule_counts(rule, table)
    rule["confidence"] = Nboth / Nleft
    rule["support"] = Nboth / Ntotal
    rule["completeness"] = Nboth / Nright
    rule['lift'] = get_support(rule['rhs'] + rule['lhs'], table) / (get_support(rule['rhs'], table) * get_support(rule['lhs'], table))

def compute_unique_values(table):
    '''Computes unique values from the table
    Args:
        table (list of list): table being mined
    '''
    unique = set()
    for row in table:
        for value in row:
            unique.add(value)
    return sorted(list(unique))

def all_rows_same_length(table):
    '''Checks if all rows are the same length
    Args:
        table (list of list): table being mined
    '''
    row_lens = [len(x) for x in table]
    if len(list(set(row_lens))) == 1:
        return True
    else:
        return False

def remove_duplicates(nested_list):
    '''Removes duplicates from a nested list (list of lists or list of sets) '''
    new_list = []
    for item in nested_list:
        if item not in new_list:
            new_list.append(item)
    return new_list

def get_union(itemSet, length):
    '''Gets all unions of sets in the itemset of the specified length '''
    all_sets = [list(set(i).union(set(j))) for i in itemSet for j in itemSet
                if len(list(set(i).union(set(j)))) == length]
    return remove_duplicates(all_sets)

def get_powerset(itemset):
    '''Returns the powerset of the inputted itemset '''
    powerset = []
    for i in range(len(itemset) + 1):
        powerset.extend([list(x) for x in list(itertools.combinations(itemset, i))])
    return powerset

def get_support(item_set, table):
    '''Gets the support of a single set (item_set) in the table '''
    cnt = 0
    for row in table:
        cnt += check_row_match(item_set, row)
    sup = cnt / len(table)
    return sup

def get_min_sup(all_sets, table, minsup):
    '''Returns only sets in all_sets that have a support above the minimum support '''
    min_sup_sets = []
    for item_set in all_sets:
        sup = get_support(item_set, table)
        if sup >= minsup:
            min_sup_sets.append(item_set)
    return min_sup_sets

def pruning(Ck, prev_L, prev_k):
    '''Removes members of Ck that have a subset of length prev_k that is not in
    the previous Lk set'''
    tempCk = Ck.copy()
    for item in Ck:
        subsets = [list(x) for x in list(itertools.combinations(item, prev_k))]
        # print('subsets', subsets)
        for subset in subsets:
            # if the subset is not in previous K-frequent get, then remove the set
            if set(subset) not in [set(x) for x in prev_L]:
                tempCk.remove(item)
                break
    return tempCk

def generate_apriori_rules(table, supported_itemsets, minconf):
    '''Checks that rules satisfy the minimum confidence '''
    rules = []
    # TODO
    # for each itemset S in supported_itemsets:
    for itemset in supported_itemsets:
    # generate all confident rules:
        possible_rhs = get_powerset(itemset)
        # start with 1 term RHSs... then 2 term RHS... len(S)-1 term RHS...
        # all items not in the RHS are in the LHS
        for rhs in possible_rhs:
            lhs = [item for item in itemset if item not in rhs]
            if rhs and lhs: #make sure neither sets are empty
                rule = {'lhs':lhs, 'rhs':rhs}

                # compute confidence, if confidence >= minconf, add to rules
                compute_rule_interestingness(rule, table)
                if rule['confidence'] >= minconf:
                    rules.append(rule)
    return remove_duplicates(rules)

def apriori(table, minsup, minconf):
    '''Apriori algorithm for associative rule generation '''
    # goal is to return a list of supported and confident rules
    # TODO
    L = {} # holds all item sets
    # supported_itemsets = [] # L2 U ... Lk-2
    # 1. create L1 the set of supported itemsets of size 1
    L[1] = [[x] for x in compute_unique_values(table)]
    # print(L[1])
    # 2. k = 2
    k = 2 # cardinality of the itemsets

    # 3. while (Lkminus1 is not empty)
    while L[k-1]: # empty lists evaluate to None
        # 4., 5., 6.,...
        Ck = get_union(L[k-1], k)
        # print('Ck', Ck)
        Ck = pruning(Ck, L[k-1], k - 1) # remove all sets from ck with subsets not in the prev L
        # print('Ck after pruning', Ck)
        L[k] = get_min_sup(Ck, table, minsup)
        # print('Lk', L[k])
        k += 1

    # union of the supported itemsets
    supported_itemsets = []
    for i in range(2, k-1):
        supported_itemsets.extend(L[i])
    # print('supported sets without duplicate sets', remove_duplicates([set(x) for x in supported_itemsets]))
    # make sure there are no duplicate sets
    supported_itemsets = [list(j) for j in remove_duplicates([set(x) for x in supported_itemsets])]
    # print('supported itemsets', supported_itemsets)
    rules = generate_apriori_rules(table, supported_itemsets, minconf)
    return rules

def compute_bootstrapped_sample(X_train, y_train, seed = None):
    if seed is not None:
        random.seed(seed)
    n = len(X_train)
    X_sample = []
    y_sample = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        X_sample.append(X_train[rand_index])
        y_sample.append(y_train[rand_index])
    return X_sample, y_sample


def compute_random_subset(values, num_values, seed = None):
    if seed is not None:
        random.seed(seed)
    shuffled = values[:] # shallow copy
    random.shuffle(shuffled)
    return sorted(shuffled[:num_values])

def calculate_covariance(x, y):
    '''Calculates the covariance between 2 datasets
    Args:
        x (list of floats):
    '''
    x_mean = calculate_mean(x)
    y_mean = calculate_mean(y)
    numerator = sum([(x[i] - x_mean)*(y[i] - y_mean) for i in range(len(x))])
    cov = numerator / len(x)
    return cov

def calculate_correlation(x, y):
    '''Returns the correlation coefficient or r-value for 2 numerical datasets
    Args:
        x (list of floats): independent variable
        y (list of floats): dependent variable
    Returns:
        r (float): r-value or correlation coefficient
    '''
    x_mean = calculate_mean(x)
    y_mean = calculate_mean(y)
    numerator = sum([(x[i] - x_mean)*(y[i] - y_mean) for i in range(len(x))])
    sum_squares_x = sum([(x[i] - x_mean) ** 2 for i in range(len(x))])
    sum_squares_y = sum([(y[i] - y_mean) ** 2 for i in range(len(y))])
    r = numerator / math.sqrt(sum_squares_x * sum_squares_y)
    return r

def compute_unique_list(list):
    '''Computes unique values from the list
    Args:
        table (list of list): table being mined
    '''
    unique_list = []
    for value in list:
        if value not in unique_list:
            unique_list.append(value)
    return unique_list

def compute_votes(col_list):
    """Gets a list of columns and is able to place into bins 
    # [27.0, 14040.5, 28054.0, 944860.0, 1861666.0]
    # [min, median(min to median), median, median(median to max), max]
    Attributes:
        col_list(list or int): A list of column values
    Returns: 
        new_list(list): new list of ratings 
        
    """
    new_list = []
    for value in col_list:
        if value >= 1861667.0:
            new_list.append(6)
        elif value >= 944861.0:
            new_list.append(5)
        elif value >= 28055.0:
            new_list.append(4)
        elif value >= 14041:
            new_list.append(3)
        elif value >= 27.0: 
            new_list.append(2)
        elif value < 27.0:
            new_list.append(1)
    return new_list
    
def normalize(self, new_instances):
        '''Normalizes any inputted data that resembles the original instances
            using the min and max values for each attribute
        '''
        assert len(new_instances[0]) == len(self.orig_instances[0])
        # now scale each attribute according the min and max values
        for i in range(len(self.attrs)):
            for j in range(len(new_instances)):
                normalized_val = (new_instances[j][i] - self.mins[i]) / (self.maxs[i] - self.mins[i])
                new_instances[j][i] = normalized_val

        return new_instances