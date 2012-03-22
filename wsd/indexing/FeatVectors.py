import WordMap
import pickle
import re

class FeatVectors:

    def __init__(self, inputfile = ""):
        try:
            self.word_map = pickle.load(open("../../data/wordmap.pkl"))
        except EOFError:
            self.word_map = WordMap.WordMap()

    def open_file(self, file_name = "../Data/practice.data.txt"):
        self.FILE = open(file_name)

    def get_file(self):
        return self.FILE

    def write_map(self):
        """
        Writes the current wordmap to a pickle file for future use
        """
        pickle.dump(self.word_map, open("../../data/wordmap.pkl","w"))

    def map_file(self, file_lines):
        """
        Takes a file object and maps out each individual line in
        the file to the map format. 

        returns an array of dictionarys containing each line object

        param 
        -----
        file_lines: A file object or list of strings with lines of the format
            "word.pos t0 t1 ... tk @ context @ target @ context" to be mapped
        """
        file_map = []

        for line in file_lines:
            file_map.append(self.map_line(line))

        return file_map

    def map_line(self, line):
        """
        Takes an individual line and parse out the components of the line 
        given the format "word.pos t0 t1 ... tk @ context @ target @ context"
        
        returns a dictionary containing those individual elements

        param 
        -----
        line: An individual line containing WSD information
        """
        line_map = {}

        line_map['word']     = line[0 : line.find('.')]
        line_map['pos']      = line[line.find('.') + 1 : line.find(' ')]
        line_map['sense']    = line[line.find(" ") + 1 : line.find("@")]
        line_map['context']  = context = line[line.find("@") + 1:]
        line_map['coll']     = coll = self.find_coll(context)
        line_map['coll_map'] = self.map_coll(coll)

        return line_map

    def find_coll(self, context, dist = 2):
        """
        Finds the collocated words surrounding the @target@ word 
        in the context

        returns a string of space separated collocations.

        param
        -----
        context: The context string surrounding the target word
        dist: The number of words leading and trailing to base
            collocations off of
        """
        if self.coll_dist:
            dist = self.coll_dist

        word_list = context.split(' ')
        
        left = right = dist

        pattern = re.compile('(@?[a-zA-Z]+@)')

        i = 0
        for word in word_list:
            #if word[0] == '@' and word[len(word)] == '@':
            if pattern.match(word):
                word_list.pop(i)
                break
            i += 1

        # Check to see if the collocation distance extends beyond the
        # size of the word list
        # Should we handle them like this or replacing empty spaces with 0s
        listlen = len(word_list)
        if i < left:
            left = i
            right = right + i if not (right + i) > listlen else listlen - i
        if i + right > listlen:
            diff = listlen - i
            right = diff
            left = left + diff if not (left + diff) > i else i

        return " ".join(word_list[i-left : i+right])

    def map_coll(self, coll):
        """
        Takes the collocation string and maps the selected term to
        indexed numbers that are unique for each word type

        param 
        -----
        coll: The collocations string with words separated
            by spaces
        """
        words = coll.split()
        nums = []

        for word in words:
            nums.append(str(self.word_map.get(word)))

        return " ".join(nums)

    def find_word_lines(self, file_lines, word):
        """
        Finds all the lines in a given file object that start with 
        the supplied word returns a list containing those lines

        returns a list of strings containing the word sense information

        param 
        -----
        file_lines: A file or list of strings to fetch the congruent 
            word information from 

        word: The word and part of speach to search and build the list 
            on. Format as <word>.pos, e.g. begin.v
        """
        lines = []

        for line in file_lines:
            if line.startswith(word):
                lines.append(line)

        return lines

    def dis_word(self, word):
        """
        Looks for a specific word and extracts the lines that are associated
        with that word and performs the mapping for each line that is 
        found

        Returns an array of dictionaries containing the sense information
        for the given word

        param
        -----
        word: The word and part of speach to search and build the list 
            on. Format as <word>.pos, e.g. begin.v
        """
        wFile = self.find_word_lines(self.get_file(), word)
        wMap = self.map_file(wFile)

        return wMap

    def dis_file(self, word, fileinput, fileoutput, coll_dist = 2):
        """
        Creates the feature vectors for a file by mapping the words 
        indexes and outputs a file that can be read by datautil.py
        to create an arff file for Machine learning

        param
        -----
        word: The string word to build the feature vectors across the
            file
        fileinput: The location of the word sense data with each line
            in the format "word.pos t0 t1 ... tk @ context @ target @ context"
        fileoutput: the location of the file to output to the for 
            the input file for Learn.java
        coll_dist: The number of previous and post collocated words 
            to get when looking at the context of a word. 
        """
        self.coll_dist = coll_dist
        
        word_lines = self.find_word_lines(open(fileinput), word)
        fileoutput = open(fileoutput, 'w')

        fmap = self.map_file(word_lines)
        
        if len(fmap) == 0:
            exit("No lines were found for that word")

        feature_count = coll_dist * 2
        class_count = len(fmap[0]['sense'].strip().split(" "))

        file_lines = []

        file_lines = self.format_headers(file_lines, feature_count, class_count)
        
        for item in fmap:
            file_lines.append(self.format_line(item))

        for line in file_lines:
            fileoutput.write(line.lstrip())
        
        self.write_map()


    def format_headers(self, file_list, feature_count, class_count):
        """
        Formats the headers of the list to be written to the output
        file that will be converted to an arff file by the datautil.py
        utility

        param
        -----
        file_list: The list of strings to be written to the file - should
            be empty when used here
        feature_count: The number of features being used in the learning
            algorithm. E.G. if we're using the 4 nearest words, then this
            would equal 4
        class_count: the number of possible classes for the word, i.e. 
            the number of possible senses for a word. 
        """
        file_list.append("# Feature Count\n")
        file_list.append(str(feature_count) + "\n")
        file_list.append("# Class Count \n")
        file_list.append(str(class_count)+ "\n")

        return file_list

    def format_line(self, line_map):
        """
        Takes a mapped line dictionary and formats it into a string
        to be written to an arff file. Writes multiple lines if the
        word contains multiple senses. 

        param
        -----
        line_map: A mapped line dictionary object in the format 
            created by the function map_line
        """
        line = []

        senses = self.map_sense(line_map['sense'])

        for sense in senses:
            # We don't need the word at the beginning of the line
            # at the moment
            #line.append(line_map['word'] + "." + line_map['pos'])
            line.append(line_map['coll_map'])
            line.append(sense)
            line.append("\n")

        return " ".join(line)

    def map_sense(self, sense):
        """
        Takes a sense string and returns a list of the senses that are
        contained within it, so with an input of "0 1 0 0 0" it will 
        return a list equal to ["2"], and for "0 1 0 1 0" it will 
        return ["2","4"]

        param
        -----
        sense: the string of senses that apply to this word in binary format
        """
        sense_list = sense.strip().split(" ")
        senses = []
        for i in range(len(sense_list)):
            if sense_list[i] == "1":
                senses.append(str(i + 1))

        return senses


