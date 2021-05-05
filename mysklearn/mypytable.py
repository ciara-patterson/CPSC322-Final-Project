##############################################
# Programmer: Ciara Patterson
# Class: CPSC 322-02, Spring 2021
# Programming Assignment #3
# 2/25/2021
#
#
# MyPyTable class implements a table object
# that we can use for processing and manipulating tabular data
# No bonus is attempted.
##############################################

import copy
import csv
import mysklearn.myutils as myutils


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        N = len(self.data)
        M = len(self.column_names) # assumes all rows are the same length
        return N, M

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # check that col_identifier is valid
        if isinstance(col_identifier, str):
            # if col_identifier is a string, check it is in column_names
            if col_identifier in self.column_names:
                col_idx = self.column_names.index(col_identifier)
            else:
                raise ValueError
        elif isinstance(col_identifier, int):
            # if col_identifier is an int, check it does not exceed num of columns
            if col_idx < len(self.column_names):
                col_idx = col_identifier
            else:
                raise ValueError
        # if col_identifier is not a str or int, it is invalid
        else:
            raise ValueError

        # get the column as a list, including missing values if specified
        if include_missing_values:
            col_vals = [x[col_idx] for x in self.data]
        else:
            col_vals = [x[col_idx] for x in self.data if x[col_idx] != 'NA']
        return col_vals

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        # for every value in the table
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                # try to convert it to a float
                try:
                    numeric_val = float(self.data[i][j])
                    self.data[i][j] = numeric_val
                # if the conversion fails
                # leave value as is and continue to next value
                except ValueError:
                    continue

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        for row in rows_to_drop:
            if row in self.data:
                self.data.remove(row)

    def load_from_file(self, filename, encode = 'utf-8'):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        self.data = []
        # load file using csv module
        with open(filename, newline = '', encoding = encode) as table_file:
            csv_obj = csv.reader(table_file, delimiter = ',')
            # append all rows to self.data
            for row in csv_obj:
                self.data.append(row)
        # store the header and remove it from the table data (self.data)
        self.column_names = self.data[0]
        self.data.pop(0)
        # convert all the data to numeric types (if applicable)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(self.column_names)
            for row in self.data:
                csv_writer.writerow(row)


    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns:
            list of list of obj: list of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        # get all column indices
        col_idxs = []
        for col in key_column_names:
            col_idxs.append(self.column_names.index(col))

        # create a new table with only relevant information
        search_table = [[x for x in row if row.index(x) in col_idxs] for row in self.data]

        # get a list of all duplicate rows
        duplicate_idxs = []
        for i in range(0, len(search_table)):
            for j in range(i + 1, len(search_table)):
                if search_table[i] == search_table[j] and j not in duplicate_idxs:
                    duplicate_idxs.append(j)

        # use indexes to get a final list
        final_list = []
        for row_idx in duplicate_idxs:
            final_list.append(self.data[row_idx])
        return final_list

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        self.data = [x for x in self.data if "NA" not in x]
        return
    
    def remove_rows_with_0_values(self, column):
        """Remove rows from the table data that contain a zero value (0).
        """
        col_idx = self.column_names.index(column)
        self.data = [x for x in self.data if 0 != x[col_idx]]
        return


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        for i in range(len(self.column_names)):
            col_vals = self.get_column(self.column_names[i],
                                        include_missing_values = False) # used for computing avg
            # check if the column value is a continuous value
            if isinstance(col_vals[0], float):
                # compute avg of column
                avg = sum(col_vals) / len(col_vals)
                # iterate through rows to replace missing items with avg
                for j in range(len(self.data)):
                    if self.data[j][i] == 'NA':
                        self.data[j][i] = avg
            else:
                continue

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """
        # build 2D list with summary stats
        table = []
        for col in col_names:
            col_vals = self.get_column(col, include_missing_values = False)
            if col_vals and isinstance(col_vals[0], float):
                row = []
                row.append(col) # attribute

                # get the min and max
                col_min = min(col_vals)
                col_max = max(col_vals)
                row.append(col_min)
                row.append(col_max)

                # calculate mid
                mid = (col_min + col_max) / 2.0
                row.append(mid)

                # calculate the mean
                mean = sum(col_vals) / len(col_vals)
                row.append(mean)

                # calculate the median
                col_vals.sort()
                mid_idx = len(col_vals) // 2
                median = (col_vals[mid_idx] + col_vals[~mid_idx]) / 2
                row.append(median)
                table.append(row)
            else:
                continue
        columns = ["attribute", "min", "max", "mid", "avg", "median"]
        return MyPyTable(columns, table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        table = []
        for row1 in self.data:
            for row2 in other_table.data:
                # check if the row in the left table matches the right table
                # by checking if the key values in each row are identical
                match = True
                for col in key_column_names:
                    idx1 = self.column_names.index(col)
                    idx2 = other_table.column_names.index(col)
                    if row1[idx1] != row2[idx2]:
                        match = False
                # only append the data to the joined table if it's a match
                if match:
                    row2_keys = [other_table.column_names.index(col) for col in key_column_names]
                    all_vals = row1 + [x for x in row2 if row2.index(x) not in row2_keys]
                    table.append(all_vals)
        # combine the columns from both tables to get the new joined table columns
        columns = self.column_names + [x for x in other_table.column_names if x not in self.column_names]
        return MyPyTable(columns, table)

    def get_key_columns(self, key_column_names):
        self_key_idxs = []
        for col in key_column_names:
            self_key_idxs.append(self.column_names.index(col))
            
        self_key_vals = []
        for row in self.data:
            new_row = [row[self.column_names.index(col)] for col in key_column_names]
            self_key_vals.append(new_row)
        return self_key_vals
        
        

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # get the index for the keys in the left table (self.data)
        self_key_idxs = []
        for col in key_column_names:
            self_key_idxs.append(self.column_names.index(col))

        # get the index for the keys in the right table (other_table)
        other_key_idxs = []
        for col in key_column_names:
            other_key_idxs.append(other_table.column_names.index(col))

        # get a table with just the key values in each table
        # these tables will be used to check if there's a match


        self_key_vals = []
        for row in self.data:
            new_row = [row[self.column_names.index(col)] for col in key_column_names]
            self_key_vals.append(new_row)


        other_key_vals = []
        for row in other_table.data:
            new_row = [row[other_table.column_names.index(col)] for col in key_column_names]
            other_key_vals.append(new_row)


        # get a list of all the columns that will be used in the final joined table
        joined_header = copy.deepcopy(self.column_names)
        joined_header.extend([col_name for col_name in other_table.column_names if col_name not in key_column_names])


        joined_table = []
        # for every row in the left table
        for row_idx1 in range(0, len(self_key_vals)):
            join_row = self.data[row_idx1]
            self_key = self_key_vals[row_idx1]
            # if the key for this row is in the other table
            if self_key in other_key_vals:
                # get the row num of the corresponding row in the other table
                other_row_idx = other_key_vals.index(self_key)
                # iterate through every value in that row and append the other_table
                # data if that value is not a key value
                for other_col_idx in range(0, len(other_table.data[other_row_idx])):
                    if other_col_idx not in other_key_idxs:
                        join_row.append(other_table.data[other_row_idx][other_col_idx])
            # if the row in the left table does not correspond to a row in the right
            # appened NA values
            else:
                for col in range(len(join_row), len(joined_header)):
                    join_row.append("NA")
            # add row to the new table
            joined_table.append(join_row)

        # now look for rows that are in the right table and not in the left
        for row_idx2 in range(0, len(other_key_vals)):
            # if the right row's keys are not in the left table's keys
            if not other_key_vals[row_idx2] in self_key_vals:
                # add the key columns to the joined table
                join_row = []
                key_counter = 0 # used to align key columns
                for col_idx in range(0, len(self.data[0]) - 1):
                    if col_idx in self_key_idxs:
                        join_row.append(list(other_key_vals[row_idx2])[key_counter])
                        key_counter += 1
                    # add NA for the non-key columns in the left table
                    else:
                        join_row.append("NA")
                # after adding the left table data, append the data associated with the right table
                for other_col_idx in range(0, len(other_table.data[row_idx2])):
                    if not other_col_idx in other_key_idxs:
                        join_row.append(other_table.data[row_idx2][other_col_idx])
                joined_table.append(join_row)

        return MyPyTable(joined_header, joined_table)

    def group_by(self, group_by_col_name):
        '''Returns a list of 2D tables divided by the value of group_by_col_name

        Args:
            group_by_col_name (str): column name of a column in self.data
            It's assumed that group_by_col_name is a valid column name
        Returns:
            group_names (list of str): list of unique values in the group by column
            group_by_subtables (list of list of list): list of 2D tables with the
                data for each group'''
        col = self.get_column(group_by_col_name, include_missing_values = False)
        col_index = self.column_names.index(group_by_col_name)
        # we need the unique values for group by column
        group_names = list(set(col))
        group_subtables = [[] for _ in group_names]

        # algorithm: walk through each row and assign it to the appropriate subtable based on
        # it's group_by_col_name value

        for row in self.data:
            group_by_value = row[col_index] # get row's group by value, in this case the model year in the row
            group_index = group_names.index(group_by_value) # get the index of the value in the subtable list
            group_subtables[group_index].append(row) # append to the appropriate subtable


        return group_names, group_subtables

    def group_by_count(self, group_by_col_name):
        '''Gets frequencies for the group by column using the MyPyTable group_by function

        Args:
            group_by_col_name (str): column name of a column in self.data
            It's assumed that group_by_col_name is a valid column name
        Returns:
            group_names (list of str): list of unique values in the group by column
            counts (list of ints): number of the instances associated with each group name
        '''
        group_names, group_subtables = self.group_by(group_by_col_name)
        counts = [len(x) for x in group_subtables] #get the number of rows in each subtable
        return group_names, counts

    def add_column(self, new_col_vals, new_col_name):
        '''Adds a new column with the inputted values. New col values must be the same length
          as self.data

          Args:
              new_col_vals (list): values to put into the column
              new_col_name (str): name of the new column
        '''
        if len(new_col_vals) != len(self.data):
            print('The length of new_col_vals does not match the number of rows in the data.')
            print('The table was not modified')
        for i in range(len(self.data)):
            self.data[i].append(new_col_vals[i])
        self.column_names.append(new_col_name)

    def process_list_column(self, list_col_name, delimiter = ','):
        '''Converts a list-like string column into a column with only list values

        Args:
            list_col_name (str): name of list-like column
            delimiter (str): the divider of items in the string
        '''
        list_col_idx = self.column_names.index(list_col_name)
        for i in range(len(self.data)):
            self.data[i][list_col_idx] = self.data[i][list_col_idx].split(delimiter)

    def normalize_columns(self, columns_to_norm):
        '''Normalize values in columns using min-max scaling

        Args:
            columns_to_norm (list of string): name of the columns to normalize

        '''
        for col in columns_to_norm:
            col_vals = self.get_column(col)
            norm_vals = myutils.normalize(col_vals)
            index = self.column_names.index(col)
            for i in len(self.data):
                self.data[i][index] = norm_vals[i]
                
    def convert_to_string(self, column_name):
        col_idx = self.column_names.index(column_name)
        for i in range(len(self.data)):
            self.data[i][col_idx] = str(self.data[i][col_idx])
            
    def load_from_file_no_encode(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table = []
        header = []
        infile = open(filename, "r")
        csv_reader = csv.reader(infile, delimiter=',')
        line_count = 0
        for row in csv_reader:
            rowValues = []
            if line_count == 0:
                for value in row:
                    header.append(value)
            if line_count != 0:
                for value in row:
                    rowValues.append(value)
                line_count += 1
                table.append(rowValues)
            else:
                line_count += 1
        infile.close()
        self.data = table
        self.convert_to_numeric()
        self.column_names = header
        return self 