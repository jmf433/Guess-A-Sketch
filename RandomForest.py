from sklearn import preprocessing
import torch
from ProcessData import *
import numpy as np
from pylab import *
from scipy import stats
import pickle
import multiprocessing as mp


class TreeClassifier(object):
    def __init__(self, n=250,
                 max_depth=500,
                 min_split=2,
                 max_features=500,  # total of 50x50 = 2500 features
                 bootstrap_samples=1000):
        """
        Initializes an object for the Random Forest Classification Tree.
        Inputs:
            n: the number of trees. Default is set to 250, the number of labels.
            max_depth: The maximum tree depth, default to 500.
            min_split: The minimum number of samples for the tree to split, default 2.
            max_features: The maximum number of features to consider for each split,
            default is the square root of the 50x50 image size. Uses Adaboost.
            bootstrap_samples: The number of samples to consider to train each tree.
            forestModel: trained random forest
        """
        self.n = n
        self.max_depth = max_depth
        self.min_split = min_split
        self.max_features = max_features
        self.bootstrap_samples = bootstrap_samples
        self.estimator_count = 0
        self.forestModel = None
        self.le = None

    class TreeNode(object):
        """
        Individual tree node for constructing classification trees.
        """

        def __init__(self, left, right, parent, cutoff_w, cutoff_h, cutoff_val, prediction):
            """
            Initializes a node of a random forest tree
            Inputs:
                left: left subtree (None if leaf)
                right: right subtree (None if leaf)
                parent: parent node
                cutoff_w: The dimension on which to split in the horizontal direction
                cutoff_h: The dimension on which to split in the vertical direction
                prediction: (if leaf node) the prediction at this node
            """
            self.left = left
            self.right = right
            self.parent = parent
            self.cutoff_w = cutoff_w  # feature id width
            self.cutoff_h = cutoff_h  # feature id height
            self.cutoff_val = cutoff_val
            self.prediction = prediction

    def bestSplit(self, features, labels):
        """
        Finds the best split among the features, subject to the constraints
        of the TreeClassifier object.
        Inputs:
            Features: n x w x h tensor containing n images of dimension w x h
            Labels: a tensor containing n numeric labels

        Returns:
            bestW: the index of the best width dimension to use
            bestH: the index of the best height dimension to use
            BestCut: a integer that denotes which value to split along.
            Loss: The gini impurity loss along this split
        """
        nd, w, h = features.size()
        nl = len(labels)
        assert (nd == nl)

        # need at least min_split values in order to consider splitting
        assert (nd >= self.min_split)

        bestloss = np.inf
        feature = (np.inf, np.inf)
        cut = np.inf

        # select a random image. Pick the non-white pixels as the feature
        # dimensions to analyze (at most max_features pixels)
        # This is as opposed to simply sampling max_features random pixels,
        # which may result in many pixels that are white for most images
        random_img = np.random.randint(low=0, high=nd, size=1)
        pix = torch.where(features[random_img] != 1, 1, 0)
        npix = torch.sum(pix)
        _, wd, hd = torch.nonzero(pix, as_tuple=True)

        if self.max_features < npix:
            # if we want to pick fewer pixels than the number of nonwhite pixels
            ii = np.random.randint(low=0, high=(
                npix-1), size=self.max_features)
            wd = wd[ii]
            hd = hd[ii]
        # select max_features random dimensions
        for i in range(min(self.max_features, npix)):
            # checks the gini impurity when splitting on feature wd[i],hd[i]
            # sort along selected dimension

            # This index will contain some color in a range from 0 to 1
            ii = (features[:, wd[i], hd[i]].argsort())
            fs = features[ii, wd[i], hd[i]]
            ls = labels[ii]

            # identify where the x-values are unique
            idif = np.where(np.abs(np.diff(fs, axis=0)) >
                            np.finfo(float).eps * 100)[0]

            # We calculate gini impurity as the probability of misclassification
            # in each split: 1 - sum(labels i){(count(i)/total)^2}
            totalL = 0
            totalR = nd

            countsL = np.zeros(max(labels)+1)
            countsR = np.zeros(max(labels)+1)

            pj = 0

            for v in labels:
                countsR[v] += 1

            for j in idif:
                images = fs[pj:j+1]
                ni = len(images)
                # move images to the left
                totalL += ni
                totalR -= ni

                for (k) in range(pj, j+1):
                    v = ls[k]
                    countsL[v] += 1
                    countsR[v] -= 1

                pj = j+1

                giniL = 1 - sum(np.square(countsL/totalL))
                giniR = 1 - sum(np.square(countsR/totalR))

                loss = totalL/nd*giniL + totalR/nd*giniR
                if loss < bestloss:
                    feature = (wd[i], hd[i])
                    cut = (fs[j] + fs[j+1])/2
                    bestloss = loss

        if feature == (np.inf, np.inf) or cut == np.inf:
            # all features selected were the same for all images
            return None, None, None, None

        return feature[0], feature[1], cut, bestloss

    def cart(self, features, labels, depth=np.inf):
        """
        Builds a CART tree, with maximum depth and bootstrap samples defined
        by self.

        Inputs:
            xTr: n x w x h matrix of images
            yTr: tensor of n numeric labels
            maxdepth: maximum tree depth

        Output:
            root: root of the CART decision tree
        """
        n, w, h = features.shape
        nl = len(labels)
        assert (nl == n)

        # find the most common label
        prediction, _ = torch.mode(labels, 0, keepdim=True)
        prediction = prediction.item()

        # base cases:
        # reached maximum depth
        # too few points to split
        # all labels are the same
        labelDif = labels-labels[0]
        if depth <= 1 or n < self.min_split or \
                torch.count_nonzero(labelDif) == n:
            # return a leaf
            return self.TreeNode(None, None, None, 0, 0, 0, prediction)
        else:
            # select bootstrap_samples random fata points
            count = min(self.bootstrap_samples, n)
            ii = np.random.randint(low=0, high=n, size=count)
            xTr = features[ii, :, :]
            yTr = labels[ii]
            # determine best split along these samples (note that we only look at
            # max_features splits along these samples)
            cutoffW, cutoffH, cutValue, _ = self.bestSplit(xTr, yTr)
            if cutoffW == None:
                # no split differentiated the values, return a leaf
                return self.TreeNode(None, None, None, 0, 0, 0, prediction)

            # extract left and right subtrees
            iL = features[:, cutoffW, cutoffH] <= cutValue
            xL = features[iL, :, :]
            yL = labels[iL]

            iR = features[:, cutoffW, cutoffH] > cutValue
            xR = features[iR, :, :]
            yR = labels[iR]

            if len(iL) == 0 or len(iR) == 0:
                # base case, one of the subtrees is empty
                return self.TreeNode(None, None, None, 0, 0, 0, prediction)
            else:
                left = self.cart(xL, yL, depth-1)
                right = self.cart(xR, yR, depth-1)
                root = self.TreeNode(left, right, None, cutoffW, cutoffH,
                                     cutValue, prediction)
                left.parent = root
                right.parent = root
                return root

    def evalCART(self, root, featuresTe):
        """
        Returns a prediction for each feature in featuresTe using the CART
        decision tree given by root.

        Inputs:
            root: a CART decision tree of type TreeNode
            featuresTe: an n x w x h tensor of images

        Outputs:
            labels: a length n tensor of label predictions, numerical
        """

        assert root is not None
        n, w, h = featuresTe.shape

        labels = torch.zeros(n)
        for i in range(n):
            # predict image i
            tree = root
            featurei = featuresTe[i, :, :]

            while tree.left and tree.right:
                # move to the correct subtree
                wd = tree.cutoff_w
                hd = tree.cutoff_h
                val = tree.cutoff_val
                if featurei[wd, hd] <= val:
                    tree = tree.left
                else:
                    tree = tree.right
            labels[i] = tree.prediction
        return labels

    def train_tree(self, features, labels):
        # randomly sample n datapoints
        n, w, h = features.shape
        ii = np.random.randint(0, n-1, n)
        xTr = features[ii, :, :]
        yTr = labels[ii]
        tree = self.cart(xTr, yTr, self.max_depth)
        return tree

    def forest(self, features, labels, indicies, m):
        """
        Creates a random forest with depth defined by self.max_depth.

        Inputs:
            features: n x w x h array of images
            labels: length n array of string labels
            m: number of trees

        Outputs:
            None. Modifies the self object to allow for evaluation of the 
            created forest.
        """
        le = preprocessing.LabelEncoder()
        nlabels = le.fit_transform(labels)
        labels = torch.as_tensor(nlabels)
        labels = labels[indicies]
        self.labelEncoder = le

        n, w, h = features.shape
        nl = len(labels)
        assert (n == nl)

        forest = []
        for i in range(m):
            # train trees
            tree = self.train_tree(features, labels)
            forest.append(tree)

        self.forestModel = forest
        self.le = le
        return ()

    def evalForest(self, featuresTe):
        """
        Makes a prediction for the images in featuresTe using the forest. Uses
        the most common label predicted for each image.
        Input:
            featuresTe: n x w x h array of images
        Output:
            labels: length n array of predictions (string format)
        """
        forest = self.forestModel
        print(self)
        le = self.le
        m = len(forest)
        n, w, h = featuresTe.shape

        # stores the predictions of each tree for each datapoint
        predictions = zeros((m, n))

        i = 0
        for tree in forest:
            preds = self.evalCART(tree, featuresTe)
            predictions[i, :] = preds
            i += 1

        # final predictions are the columnwise modes
        modes, counts = stats.mode(predictions, keepdims=True)

        # convert it back to a tuple of labels
        labels = le.inverse_transform(modes[0].astype(int))

        return labels


def train_ith_forest(length, label_lengths, train_features, train_labels):
    """
        Trains a size 10 forest for a given word length
        Input:
            length: the word length
            label_lengths: the lengths of all words in the training set
            train_features: n x w x h array of images to train on, may include 
                images whose labels are different lengths
            train_labels: length-n tuple containing the corresponding labels
        Output:
            length, model: the specified word length and trained RF model
    """
    indicies = label_lengths == length
    features = train_features[indicies]
    labels = train_labels

    model = TreeClassifier()
    model.forest(features, labels, indicies, 10)

    print("forest for length " + str(length.item()) + " trained")
    return length, model


def train(batch_size):
    """
        Trains a size 10 forest for images in folder NewTrainingData, given word
        length. 
        Input:
            batch_size: the number of images to train on
        Output:
            Saves the model to file called 'RandomForestModel.obj'
    """
    trainingLoader = extractData(batch_size, "training", "NewTrainingData")
    train_labels, train_features = next(iter(trainingLoader))
    train_features = train_features/255

    # We will train a random forest for each possible word length
    # This is because in our model, humans and the machine are given the
    # word length to help them guess the image.

    # determine the lengths of each label
    label_lengths = torch.zeros(len(train_labels))
    for i in range(len(train_labels)):
        label_lengths[i] = len(train_labels[i])

    # determine the unique lengths
    unique_lengths = torch.unique(label_lengths)

    # for each unique length, train a new forest
    # use concurrency
    res = {}
    pool = mp.Pool(processes=7)

    results = pool.starmap(train_ith_forest, [(
        i, label_lengths, train_features, train_labels) for i in unique_lengths])

    for length, model in results:
        res[float(length.item())] = model

    filehandler = open('RandomForestModel.obj', 'wb')
    pickle.dump(res, filehandler)
    return


def test(batch_size):
    """
        Tests the model located in file 'RandomForestModel.obj' on images from
        folder "NewTestingFolder"
        Input:
            batch_size: the number of images to test on, maximum 1250
        Output:
            Accuracy: The fraction of images that were classified correctly
    """
    testingLoader = extractData(batch_size, "testing", "NewTestingData")
    test_labels, test_features = next(iter(testingLoader))
    test_features = test_features/255

    filehandler = open("RandomForestModel.obj", 'rb')
    models = pickle.load(filehandler)

    # determine the lengths of each label
    label_lengths = torch.zeros(len(test_labels))
    for i in range(len(test_labels)):
        label_lengths[i] = len(test_labels[i])

    # determine the unique lengths
    unique_lengths = torch.unique(label_lengths)

    # evaluate each word on the model for their respective length
    labels = np.array(test_labels)
    n = 0
    ncorrect = 0
    for i in unique_lengths:
        model = models[float(i.item())]
        indicies = label_lengths == i
        features = test_features[indicies]
        pred = model.evalForest(features)
        n += len(pred)
        ncorrect += sum(labels[indicies] == pred)

    return ncorrect/n


if __name__ == '__main__':
    # train(7500)
    accuracy = test(1250)
    print(accuracy)
