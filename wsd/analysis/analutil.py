import os

def rank_attr_by_info_gain(fileinput, fileoutput, numattr):
    """
    Takes in an arff file input containing a bag-of-word attr information and
    return top 'numattr' number of attributes with the highest information gain.
    The output file is in arff format containing the information gains of the
    top 'numattr' attributes.

    -------

    inputs
        fileintput:  an arff file containing attribute information in a
	    bag-of-word format
        fileoutput:  a filename to record the result in the arff format
	numattr   :  an integer representing the top number of attributes to
	    be selected for their comparatively high information gains

    returns
        an arff file labeled 'fileoutput' containing the information gains of
	    the top 'numattr' attributes

    -------

    fileinput format:
    @relation bow-train
    @attribute 'Feature 0' numeric
    @attribute 'Feature 1' numeric
    @attribute 'Feature 2' numeric
    @attribute 'Feature 3' numeric
    @attribute 'Feature 4' numeric
    @attribute Class {begin.v,begin.n}
    @data
    -0.168,1.627,-0.388,0.529,-0.874,begin.v
    1.933,-0.363,0.51,-0.621,-0.488,begin.n

    fileoutput format:
    @relation 'bow-train-weka.filters.AttributeSelectionFilter-Eweka.attributeS
    @attribute 'Feature 1' numeric
    @attribute 'Feature 0' numeric
    @attribute 'Feature 4' numeric
    @attribute Class {begin.v,begin.n}
    @data
    -0.168,1.627,-0.67,begin.v
    -0.265,1.933,-0.694,begin.n
    """
    cmmd = 'java weka.filters.AttributeSelectionFilter -S \"weka.attributeSelection.Ranker -N ' + str(numattr) + '\" -E \"weka.attributeSelection.InfoGainAttributeEval\" -i ' + fileinput + ' -o ' + fileoutput + ' -c last'
    os.system(cmmd)
