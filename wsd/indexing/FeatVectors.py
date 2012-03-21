#import nltk
import WordMap

class FeatVectors:

    def __init__(self, inputfile = ""):
        self.word_map = WordMap.WordMap()

        if not inputfile == "":
            self.FILE = inputfile

    def open_file(self, file_name = "../Data/practice.data.txt"):
        self.FILE = open(file_name)

    def get_file(self):
        return self.FILE

    def map_file(self, file_lines):
        """
        Takes a file object and maps out each individual line in
        the file to the map format. 

        returns an array of dictionarys containing each line object

        param file_lines: A file object or list of strings with lines of the format
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

        param line: An individual line containing WSD information
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
        context: The context string surrounding the target word
        dist: The number of words leading and trailing to base
            collocations off of
        """
        word_list = context.split(' ')

        left = right = dist

        i = 0
        for word in word_list:
            if word[0] == '@' and word[-1] == '@':
                word_list.pop(i)
                break
            i += 1

        # Check to see if the collocation distance extends beyond the
        # size of the word list
        listlen = len(word_list)
        if i < left:
            left = i
            right = dist + i if not (dist + i) > listlen else listlen - i
        if i + right > listlen:
            diff = listlen - i
            right = diff
            left = left + diff if not (left + diff) > i else i

        return " ".join(word_list[i-left : i+right])

    def map_coll(self, coll):
        """
        Takes the collocation string and maps the selected term to
        indexed numbers that are unique for each word type

        param coll: The collocations string with words separated
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
        word: The word and part of speach to search and build the list 
            on. Format as <word>.pos, e.g. begin.v
        """
        wFile = self.find_word_lines(self.get_file(), word)
        wMap = self.map_file(wFile)

        return wMap
