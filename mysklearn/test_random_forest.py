import numpy as np
import scipy.stats as stats
from mysklearn.myutils import MinMaxScale
from mysklearn.myclassifiers import MyRandomForestClassifier

## DECISION TREE TEST DATA

# in class decision tree example data
header = ["level", "lang", "tweets", "phd"]
attribute_domains = {"level": ["Senior", "Mid", "Junior"],
        "lang": ["R", "Python", "Java"],
        "tweets": ["yes", "no"],
        "phd": ["yes", "no"]}

X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

y_train = ["False", "False", "True", "True", "True", "False", "True", "False",
"True", "True", "True", "True", "True", "False"]

interview_tree = \
        ["Attribute", "att0",
            ["Value", "Junior",
                ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]
X_test_interview = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
y_test_interview = ['True', 'False']


## resulting tree for the N = 2, M = 1, F = 1 test
test_tree = ['Attribute', 'att0',
                ['Value', 'no',
                    ['Leaf', 'True', 5, 14]],
                ['Value', 'yes',
                    ['Leaf', 'False', 9, 14]]]

# resulting trees for the N = 3, M = 2, F = 2 test
test_forest =  [['Attribute', 'att0', # ROOT OF TREE 1
                    ['Value', 'Senior',
                        ['Attribute', 'att1',
                            ['Value', 'no',
                                ['Leaf', 'False', 1, 4]],
                            ['Value', 'yes',
                                ['Leaf', 'False', 3, 4]]]],
                    ['Value', 'Mid',
                        ['Leaf', 'True', 3, 14]],
                    ['Value', 'Junior',
                        ['Attribute', 'att1',
                            ['Value', 'no', ['Leaf', 'True', 2, 7]],
                            ['Value', 'yes', ['Leaf', 'False', 5, 7]]]]],
                ['Attribute', 'att0', # ROOT OF TREE 2
                    ['Value', 'Senior',
                        ['Attribute', 'att1',
                            ['Value', 'Java',
                                ['Leaf', 'False', 4, 6]],
                            ['Value', 'Python',
                                ['Leaf', 'False', 1, 6]],
                            ['Value', 'R',
                                ['Leaf', 'True', 1, 6]]]],
                    ['Value', 'Mid',
                        ['Leaf', 'True', 5, 14]],
                    ['Value', 'Junior', ['Leaf', 'True', 3, 14]]]]


def check_tree_equivalence(tree1, tree2):
    '''Recursively checks if trees are equal without regard for the order of nodes
    on the same level of the tree

    Args:
        tree1 (list of lists): nested list representation of a decision tree
        tree2 (list of lists): nested list representation of another decision tree
    '''

    if tree1[0] == "Value":
        check_tree_equivalence(tree1[2], tree2[2])
    elif tree1[0] == "Leaf":
        assert tree1 == tree2
    elif tree1[0] == "Attribute":
        assert tree1[1] == tree2[1]
        assert len(tree1) == len(tree2)
        for sub_num in range(2, len(tree1)):
            value = tree1[sub_num][1]
            # find the index with the corresponding value in the other tree
            # fail the test if the value is not on this level of edges
            sub_ind = -1
            for sub_tree in range(2, len(tree2)):
                if tree2[sub_tree][1] == value:
                    sub_ind = sub_tree
            assert sub_ind != -1
            check_tree_equivalence(tree1[sub_num], tree2[sub_ind])
    else:
        assert False == True # anything else should stop the test

def test_random_forest_fit():
    # test on the interview dataset

    # test 1
    N = 2
    M = 1
    F = 1
    trees = MyRandomForestClassifier(N = N, M = M, F = F, seed = 0)
    trees.fit(X_train, y_train)
    assert len(trees.learners) == M
    best_tree = [x.tree for x in trees.learners][0]
    check_tree_equivalence(test_tree, best_tree)

    # test 2
    N = 3
    M = 2
    F = 2
    trees = MyRandomForestClassifier(N = N, M = M, F = F, seed = 0)
    trees.fit(X_train, y_train)
    # print([x.tree for x in trees.learners])
    assert len(trees.learners) == M
    tree_results = [x.tree for x in trees.learners]
    for i in range(len(tree_results)):
        check_tree_equivalence(tree_results[i], test_forest[i])
