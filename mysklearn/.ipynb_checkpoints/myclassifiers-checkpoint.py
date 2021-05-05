import mysklearn.myutils as myutils
import random
from collections import Counter
import copy

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        temp = []
        flatten_X_train = myutils.flatten_list(X_train)
        self.slope, self.intercept = myutils.calculate_slope(flatten_X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        flatten_X_test = myutils.flatten_list(X_test)
        for x in flatten_X_test:
            y_pred = (x * self.slope) + self.intercept
            y_predicted.append(y_pred)
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        # compute distances for all of the X_test samples
        distances = []
        neighbor_indices = []
        for x_test in X_test:
            # for each item in the train set, compute the euclidean distances
            dists = []
            indices = []
            for i, instance in enumerate(self.X_train):
                indices.append(i)
                dists.append(myutils.compute_euclidean_distance(self.X_train[i], x_test))

            # sort the dists and move the indices to match the sorted list
            # by combining the two lists into a list of tuples, sorting, and unpacking
            sorted_dists, sorted_indices = (list(tup) for tup in zip(*sorted(zip(dists, indices))))

            # slice the lists to only include k neighbors and append to the test lists
            distances.append(sorted_dists[:self.n_neighbors])
            neighbor_indices.append(sorted_indices[:self.n_neighbors])
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, neighbor_indices = self.kneighbors(X_test)
        y_predicted = []
        for i in range(len(neighbor_indices)):
            neighbor_labels = [self.y_train[i] for i in neighbor_indices[i]]
            majority = Counter(neighbor_labels).most_common(1)
            majority_label = list(majority[0])[0]
            y_predicted.append(majority_label)
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        priors(dict of item:float pairs): The prior probabilities computed for each
            label in the training set.
        posteriors(dict of item:dict pair): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None
        self.y_train = None
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # compute the priors (probability of each class in the dataset)

        # get the number of instances for each class
        self.priors = {}
        for y in y_train:
            if y in self.priors.keys():
                self.priors[y] += 1
            else:
                self.priors[y] = 1

        # compute posteriors for each class
        self.posteriors = {}
        for y in self.priors.keys():
            self.posteriors[y] = {} # holds the posteriors for each class
            denom = self.priors[y] # get the denominator for this class
            num_attrs = len(X_train[0])
            # get just the rows corresponding to that class
            y_rows = [X_train[i] for i in range(len(y_train)) if y_train[i] == y]
            for i in range(num_attrs):
                for x in y_rows:
                    if (i, x[i]) in self.posteriors[y].keys():
                        self.posteriors[y][(i, x[i])] += 1
                    else:
                        self.posteriors[y][(i, x[i])] = 1

            # divide the numbers of each attribute by the number of instances of the class
            for key in self.posteriors[y].keys():
                self.posteriors[y][key] = self.posteriors[y][key] / len(y_rows)



        # divide by the number of instances to get the probability
        for y in self.priors.keys():
            self.priors[y] = self.priors[y] / len(y_train)




    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        labels = list(self.posteriors.keys())
        for x in X_test: # for test instance
            probs = [] # holds probability of each class, parallel to labels
            for label in labels:
                label_prob = 1
                for attr in range(len(x)):
                    if (attr, x[attr]) in self.posteriors[label].keys():
                        label_prob *= self.posteriors[label][(attr, x[attr])]
                    else:
                        label_prob = 0 # cannot make a prediction based on the data
                label_prob *= self.priors[label] # multiply by overall class probability
                probs.append(label_prob)
            # now use probabilities to make a prediction for each class
            pred = labels[probs.index(max(probs))] # get index with highest probability
            y_predicted.append(pred)

        return y_predicted # TODO: fix this

class ZeroRClassifier:
    '''Classifier that always predicts the most common label in the training set

    Attributes:
        y_train (list of items): list of labels in the training set
        most_common (item): most common label in the training set
    '''
    def __init__(self):
        '''Initializer for ZeroRClassifier'''
        self.X_train = None
        self.y_train = None
        self.most_common = None

    def fit(self, X_train, y_train):
        '''Gets the most common label

        Args:
            y_train (list of items): list of labels in the training set
        '''
        self.y_train = y_train
        self.X_train = X_train
        majority = Counter(self.y_train).most_common(1)
        self.most_common = list(majority[0])[0]

    def predict(self, X_test):
        '''Returns the most common label as the prediction for every item in the test set

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        '''
        return [self.most_common for _ in X_test]

class RandomClassifier:
    '''Classifies an item according to a weighted random choice in the training set

    Attributes:
        priors(dict of item:float pairs): The prior probabilities computed for each
            label in the training set.
    '''
    def __init__(self):
        self.priors = None
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        '''Gets the probability of each label in the test set

        Args:
            y_train (list of items): list of labels in the training set
        '''
        self.X_train = X_train
        self.y_train = y_train
        # get the number of instances for each class
        self.priors = {}
        for y in y_train:
            if y in self.priors.keys():
                self.priors[y] += 1
            else:
                self.priors[y] = 1
        # divide by the number of instances to get the probability
        for y in self.priors.keys():
            self.priors[y] = self.priors[y] / len(y_train)

    def predict(self, X_test):
        '''Returns random choice as the prediction for every item in the test set
        Choices are weighted according to the number of appearances of the label in the test set

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        '''
        y_choices = list(self.priors.keys())
        y_weights = tuple(self.priors.values())
        random_preds = random.choices(y_choices, weights = y_weights, k = len(X_test))
        return random_preds

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
        header (list of str): computed header for the training data
        attribute domains (dict): dict of unique value associated with each attribute

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domains = None



    def tdidt(self, current_instances, available_attributes):

        # select an attribute to split on
        split_attribute = myutils.select_attribute(current_instances, available_attributes,
                                                    self.attribute_domains, self.header)
        available_attributes.remove(split_attribute)

        # cannot split on the same attribute twice in a branch
        # recall: python is pass by object reference!!
        tree = ["Attribute", split_attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = myutils.partition_instances(current_instances, split_attribute,
                                                self.attribute_domains, self.header)

        # for each partition, repeat unless one of the following occurs (base case)
        for attribute_value, partition in partitions.items():
            value_subtree = ["Value", attribute_value]

            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(partition) > 0 and myutils.all_same_class(partition):
                leaf_node = myutils.create_leaf_node(partition, partitions, case = 1)
                value_subtree.append(leaf_node)
                tree.append(value_subtree)

            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(partition) > 0 and len(available_attributes) == 0:
                leaf_node = myutils.create_leaf_node(partition, partitions, case = 2)
                value_subtree.append(leaf_node)
                tree.append(value_subtree)

            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(partition) == 0:
                # replace attribute node with majority vote leaf node
                # tree = create_leaf_node(partition, partitions, case = 3)
                # break # don't look at the other attributes because we are replacing attribute node with leaf node
                return None

            else: # all base cases are false... recurse!!
                subtree = self.tdidt(partition, available_attributes.copy())
                # check if previous step was a case 3
                # create a majority vote node if so
                if subtree is None:
                    leaf_node = myutils.create_leaf_node(partition, partitions, case = 2)
                    value_subtree.append(leaf_node)
                else:
                    # need to append subtree to value_subtree and appropriately append value subtre
                    # to tree
                    # subtree is 3rd value in list
                    value_subtree.append(subtree)
                tree.append(value_subtree)

        return tree


    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.header = ["att" + str(i) for i in range(len(self.X_train[0]))]
        self.attribute_domains = myutils.compute_attribute_domains(self.X_train, self.header)
        # TODO: compute a "header" ["att0", "att1", ...]
        # my advice is to stitch together X_train and y_train
        train = [self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))]
        # initial call to tdidt current instances is the whole table (train)
        available_attributes = self.header.copy() # python is pass object reference
        self.tree = self.tdidt(train, available_attributes)

    def tdidt_predict(self, tree, instance):
        '''Uses recursion to predict the class label for an instance

        Args:
            instance (list of objs): an instance whose label is to predicted
        '''
        node_type = tree[0]
        if node_type == "Attribute":
            attribute_index = self.header.index(tree[1])
            test_value = instance[attribute_index]
            for i in range(2, len(tree)):
                value_list = tree[i]
                if value_list[1] == test_value:
                    return self.tdidt_predict(value_list[2], instance)
        else: # node_type == "Leaf"
            leaf_label = tree[1]
            return leaf_label

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for x in X_test:
            label = self.tdidt_predict(self.tree, x)
            y_predicted.append(label)
        return y_predicted # TODO: fix this

    def decision_rule_recursion(self, tree, rule, rules):
        '''Recursive algorithm that traverses the tree to identify rules

        Args:
            rule (list of str): list that represents a branch where each str is a node
            rules (list of list of str): list that contains all possible rules in the tree
        '''

        if tree[0] == "Leaf":
            rule.append("THEN {} = {}.".format("class", tree[1]))
            rules.append(rule)

        elif tree[0] == "Attribute":
            for i in range(2, len(tree)):
                r = list(rule) # copy the current rule
                r.append("{} == {}".format(tree[1], tree[i][1]))
                self.decision_rule_recursion(tree[i][2], r, rules)


    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rule = []
        rules = []
        self.decision_rule_recursion(self.tree, rule, rules)

        # rules holds each rule as a list, now print them one by one
        for r in rules:
            # format the values in rule appropriately
            r_str = 'IF '
            r_str += ' AND '.join(r[:-1])
            r_str += ' '
            r_str += r[-1].replace("class", class_name)

            # replace generic attribute names with real names if a list is given
            if attribute_names is not None:
                for i in range(len(attribute_names)):
                    att_str = "att" + str(i)
                    r_str = r_str.replace(att_str, attribute_names[i])
            print(r_str) # print rule

    # BONUS METHOD
    # def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
    #     """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).
    #
    #     Args:
    #         dot_fname(str): The name of the .dot output file.
    #         pdf_fname(str): The name of the .pdf output file generated from the .dot file.
    #         attribute_names(list of str or None): A list of attribute names to use in the decision rules
    #             (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
    #
    #     Notes:
    #         Graphviz: https://graphviz.org/
    #         DOT language: https://graphviz.org/doc/info/lang.html
    #         You will need to install graphviz in the Docker container as shown in class to complete this method.
    #     """
    #     pass # TODO: (BONUS) fix this

class MyRandomForestClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        N (int): The number of trees to be trained
        M (int): The number of classifiers from the set of N trees to keep (based on accuracy)
        F (int): The number of attributes to be randomly sampled from the training set
        learners (list of MyDecisionTreeClassifier): list of weak learners in the ensemble
        accuracies (list of float): accuracies of each of the learners on the validation set
            parallel to the learners list and used to select the M trees

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, N, M, F, seed):
        """Initializer for MyRandomForestClassifier.

        """
        self.X_train = None
        self.y_train = None
        self.N = N
        self.M = M
        self.F = F
        self.seed = None
        self.learners = None
        self.accuracies = 0

    def fit(self, X_train, y_train):
        ''' Fits the random forest model to a given training set

        Args:

            X_train(list of list of obj): The list of training instances (samples).
                    The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train).
                The shape of y_train is n_samples
        '''

        self.X_train = copy.deepcopy(X_train)
        self.y_train = copy.deepcopy(y_train)
        self.learners = []
        self.accuracies = []

        # generate N learners
        for i in range(self.N):
            
            # create the bootstrap sample
            if self.seed is not None:
                X_sample, y_sample = myutils.compute_bootstrapped_sample(self.X_train, self.y_train, self.seed)
            else:
                X_sample, y_sample = myutils.compute_bootstrapped_sample(self.X_train, self.y_train)


            # create the validation set
            X_val = [x for x in self.X_train if x not in X_sample]
            y_idxs = [self.X_train.index(x) for x in X_val]
            y_val = [self.y_train[idx] for idx in y_idxs]

            # get only a random subset of attributes for each sample
            values = [i for i in range(len(self.X_train[0]))] # num of items in header

            if self.seed is not None:
                F_attributes = myutils.compute_random_subset(values, self.F, self.seed)
            else:
                F_attributes = myutils.compute_random_subset(values, self.F)

            # get only those attributes from the training set
            for i in range(len(X_sample)):
                X_sample[i] = [X_sample[i][j] for j in range(len(X_sample[i])) if j in F_attributes]

            # get only those attributes from the validation set
            for i in range(len(X_val)):
                X_val[i] = [X_val[i][j] for j in range(len(X_val[i])) if j in F_attributes]


            # build a decision tree from the sample
            tree = MyDecisionTreeClassifier()
            tree.fit(X_sample, y_sample)
            self.learners.append(tree)

            # test the trees accuracy on the validation set
            y_pred = tree.predict(X_val)

            self.accuracies.append(myutils.compute_accuracy(y_pred, y_val))

        # get only the best M learners

        # sort the dists and move the indices to match the sorted list
        # by combining the two lists into a list of tuples, sorting, and unpacking
        sorted_accs, sorted_idxs = (list(tup) for tup in zip(*sorted(zip(self.accuracies, range(len(self.learners))))))

        # slice the lists to only include the M best learners
        self.learners = [self.learners[i] for i in range(len(self.learners)) if i in sorted_idxs[:self.M]]
        # self.learners = sorted_learners[:M+1]
        self.accuracies = sorted_accs[:self.M]


    def predict(self, X_test):
        ''' Predicts the class labels of a set of test instances

        Args:
            X_test (list of list of obj): The list of test instances (samples).
                    The shape of X_train is (n_train_samples, n_features)

        Returns:
            y_predicted (list of labels): labels corresponding to the test set
        '''
        # get predictions from all of the trees
        all_preds = []
        for tree in self.learners:
            preds = tree.predict(X_test)
            all_preds.append(preds)

        y_predicted = []
        # get the most common prediction for each x value
        for i in range(len(X_test)):
            x_preds = [p[i] for p in all_preds]

            # get most common prediction (majority vote)
            majority = Counter(x_preds).most_common(1)
            majority_label = list(majority[0])[0] # unpack the object to get the label
            y_predicted.append(majority_label)
        return y_predicted
