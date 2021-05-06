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

# fit the KNN algorithm to the movies data
kn_class = MyKNeighborsClassifier()
feature_cols = ['budget', 'votes', 'genre', 'rating', 'score', 'star', 'director', 'writer']
features = movies_table.get_key_columns(feature_cols)
outcomes = profitted
kn_class.fit(features, outcomes

packaged_object = kn_class

# pickle packaged object
outfile = open('movies_tree.p', 'wb')
pickle.dump(packaged_object, outfile)
outfile.close()
