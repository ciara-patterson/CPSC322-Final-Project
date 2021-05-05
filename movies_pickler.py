from mysklearn.myclassifiers import MyKNeighborsClassifier
import os
from mysklearn.mypytable import MyPyTable
import mysklearn.myevaluation as myeval
import mysklearn.myutils as myutils
import pickle

# Importing the data and table and cols
movies_fname = os.path.join("input_data", "movies.csv")
# movie_data = MyPyTable().load_from_file_no_encode(movies_fname)
movies_table = MyPyTable().load_from_file(movies_fname, encode = 'cp1252')

# Getting profit
gross_profit = [movies_table.get_column('gross')[i] - movies_table.get_column('budget')[i] for i in range(len(movies_table.data))]
profitted = [0 if gross < 0 else 1 for gross in gross_profit]
movies_table.add_column(profitted, 'profitted')

def test_folds(k, X_train_folds, X_test_folds, classifier, confusion_matrix = False):
    # evaluation metrics for each fold
    all_preds = []
    actuals = []
    for i in range(k):
        # get features and labels
        feature_cols = ['budget', 'votes', 'genre', 'rating', 'score', 'star', 'director', 'writer']
        features = movies_table.get_key_columns(feature_cols)
        outcomes = profitted

        # get the train and test set for linear_regression
        X_train = [features[j] for j in range(len(features)) if j in X_train_folds[i]]
        y_train = [outcomes[j] for j in range(len(outcomes)) if j in X_train_folds[i]]
        X_test = [features[j] for j in range(len(features)) if j in X_test_folds[i]]
        y_test = [outcomes[j] for j in range(len(outcomes)) if j in X_test_folds[i]]

        # fit the linear regression model
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        all_preds.extend(y_pred) # add the predictions
        actuals.extend(y_test) # add the actuals

    if confusion_matrix:
        # print out the confusion matrix for the labels
        labels = list(set(actuals))
        print('Confusion matrix')
        matrix = myeval.confusion_matrix(actuals, all_preds, labels)
        print(myutils.format_confusion_matrix(matrix, labels))


    else:
        # compute accuracy for the train and test sets
        acc = myutils.compute_accuracy(all_preds, actuals)
        error = myutils.compute_accuracy(all_preds, actuals, error_rate = True)

        print('accuracy = {}, error rate = {}'.format(round(acc, 2), round(error,2)))


# Splitting into folds
k = 10
X_train_folds_strat, X_test_folds_strat = myeval.stratified_kfold_cross_validation(movies_table.data, profitted, n_splits=k)

kn_class = MyKNeighborsClassifier()
test_folds(k, X_train_folds_strat, X_test_folds_strat, kn_class)

packaged_object = kn_class
# pickle packaged object
outfile = open('movies_tree.p', 'wb')
pickle.dump(packaged_object, outfile)
outfile.close()
