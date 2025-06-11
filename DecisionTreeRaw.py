import pandas as pd


computer_data = pd.read_csv("allCSV/computer_purchase.csv")
header = list(computer_data.columns)
computer_data = computer_data.values.tolist()


def unique_vals(rows, col):
    """ Find the unique value in the dataset to avoid redundency"""
    return set([row[col] for row in rows])


##########
# print(unique_vals(computer_data, 0))
# print(unique_vals(computer_data, 1))
# print(unique_vals(computer_data, -1))
##########


def countClass(rows):
    """Count the number of each type in a dataset"""
    counts = {}
    for row in rows:
        label = row[-1]  # Last column is 'Buy_Computer'
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

##########
# print(countClass(computer_data))
##########


def isNumeric(value):
    """ checks if the value is numeric basically, Int, Float or Not"""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """ this class seprates the data into small partitions"""

    def __init__(self, column, value):
        """ Simply just parse the data to class """
        self.column = column
        self.value = value

    def __repr__(self):
        """Display the question in a readable form."""
        condition = "=="
        if isNumeric(self.value):
            condition = ">="
        return f"Is {header[self.column]} {condition} {str(self.value)}?"

    def match(self, example):
        """ compare the feature value to the class value """
        val = example[self.column]
        if isNumeric(val):
            return val >= self.value
        else:
            return val == self.value



def partition(rows, question):
    """ this partitions the dataset into subsets"""

    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini_impurity(rows):
    """ this function returns a gini impurities for the row"""
    count = countClass(rows)
    impurtiy = 1
    for label in count:
        prob_of_label = count[label] / float(len(rows))
        impurtiy -= prob_of_label ** 2
    return impurtiy # returns the impurity for the rows


def info_gain(left, right, current_uncertainty):
    """ This calculates the info gain which
    ,uncertainty of starting - weight impurity of two nodes child"""
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini_impurity(left) - (1 - p) * gini_impurity(right)


def find_best_split(rows):
    """ Find the best question to ask after iterating over every possibility """
    best_gain = 0
    best_question = None
    current_uncertainty = gini_impurity(rows)
    n_features = len(rows[0])

    for col in range(n_features):
        values = set(row[col] for row in rows) # unique value in the column

        for val in values:
            question = Question(col, val) # instance of a class

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip the split if dataset doesn't divide the set
            if(len(true_rows) == 0 or len(false_rows) == 0):
                continue

            # calulate info gain for split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # if gain >= best gain then it is our new best gain obviously
            if(gain >= best_gain):
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """This is a final data we are looking for in a dataset"""

    def __init__(self, rows):
        self.predictions = countClass(rows)


class DecisionNode:
    """ This class asks the question before the actual splitting
    and will check best splitting"""

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def buildTree(rows):
    """ Builds a tree just like DSA tree """
    gain, question = find_best_split(rows)

    # Base case : No further information gain
    if gain == 0:
        return Leaf(rows)

    # Value to partition on is found here
    true_rows, false_rows = partition(rows, question)

    # Build tree from this rows
    true_branch = buildTree(true_rows)
    false_branch = buildTree(false_rows)

    # finally return the decisionNode
    return DecisionNode(question, true_branch, false_branch)


def printTree(node, spacing=""):
    """ This function will print the whole tree """

    # Base case : IF node
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # print the question to be asked for prediction
    print(spacing + str(node.question))

    # call this recursion function for true branch
    print(spacing + " ---> TRUE : ")
    printTree(node.true_branch, spacing=" ")

    # call this recursively for false branch too
    print(spacing + " ---> FALSE : ")
    printTree(node.false_branch, spacing=" ")


def classify(row, node):

    # Base case: We've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    """ Decide whether to follow a true branch or a false branch"""
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def printLeaf(counts):
    """ Prints the leaf predictions """
    total = sum(counts.values()) * 1.0
    probs = {}
    for label in counts.keys():
        probs[label] = str(int(counts[label] / total * 100)) + "%"
    return probs


def main():
    my_tree = buildTree(computer_data)
    printTree(my_tree)

    for row in computer_data:
        print("Actual: %s. Predicted: %s" % (row[-1], printLeaf(classify(row, my_tree))))

main()