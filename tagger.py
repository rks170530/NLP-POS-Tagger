import os
import io
import sys

# The string and array of the predifined the tag list.
# Set as global variable since it does not get manipulated in the program.
tagString = "NOUN, PRONOUN, VERB, ADVERB, ADJECTIVE, CONJUNCTION, PREPOSITION, DETERMINER, NUMBER, PUNCT, X"
tagArray = tagString.split(', ')

# Creates the Tagger class.
class Tagger:

    # Keeps the initial tag, transition, and emission probabilities. 
    def __init__(self):
        self.initial_tag_probability = None
        self.transition_probability = None
        self.emission_probability = None

    # Loads the Corpus' train folder.
    def load_corpus(self, path):

        # Check if the path is a directory.
        if not os.path.isdir(path):
            sys.exit("Input path is not a directory")
        
        sentenceArray = []

        # For each file in the path.
        for fileName in os.listdir(path):
            fileName = os.path.join(path, fileName)
            
            # Open and read the file.
            try:
                fileReader = io.open(fileName, 'r', encoding='utf-8', errors='ignore')

                # Until there are no more lines.
                while True:
                    
                    # Read the line.
                    wordArray = []
                    currentLine = fileReader.readline()

                    # Exit if there are no new lines.
                    if not currentLine:
                        break
                
                    # Remove end-line and convert to lowercase.
                    currentLine = currentLine.replace('\n', "")

                    # Save the line only if it isn't empty.
                    if (currentLine != ""):
                        # Append every word in the line into the word and spam array.
                        for word in currentLine.split(' '):
                            
                            # Split the word and tag.
                            try:
                                tokenTuple = word.split('/')
                                wordArray.append((tokenTuple[0], tokenTuple[1]))
                            except:
                                pass

                        # print(wordArray)
                        sentenceArray.append(wordArray)
            
            # Exit otherwise.
            except IOError:
                sys.exit("Cannot read file")

        # Return the snetenceArray.
        return sentenceArray

    # Initialize the Viterbi algorithm.
    def initialize_probabilities(self, sentences):

        # Check if the sentences is a list and output otherwise. 
        if type(sentences) != list:
            sys.exit("Incorrect input to method")


        # Create a tag dictionary.
        tagDict = {}

        # For each tag in the tagArray, add it to the tag dictionary 
        # if it does not already exist. Set it to 1 for add-one smoothing.
        for tag in tagArray:
            if tag not in tagDict:
                tagDict[tag] = 1

        # Copy the tag Dictionary to the probability dictionary.
        tagProbDict = tagDict.copy()


        # For each sentence in sentences.
        for s in sentences:

            # Get the tag of each first word in the sentence.
            wordTag = s[0][1]


            # For each word's tag in the tagArray, increment the counter in the tagProbDict.
            if wordTag in tagArray:

                tagProbDict[wordTag] += 1


        # For each tag in the tagArray, divide the count by the length (number) of sentences
        # as well as the the length of the tagArray (number of tags), for add-one smoothing.
        for tag in tagArray:
            tagProbDict[tag] = tagProbDict[tag] / (len(sentences) + len(tagArray))
        
        # Set the intial tag probability.
        self.initial_tag_probability = tagProbDict.copy()
        
        # Creat a transition probability.
        tranDictProb = {}

        # Keep track of the number of transitions and unique transitions.
        tranCount = 0
        uniqueTranCount = 0

        # For the first tag in the tag dictionary.
        for tag1 in tagDict:

            # For the second tag in the tag dictionary, create a dictionary
            # with all possible (tag1, tag2) combinations, set to 1 for 
            # add-one smoothing.
            for tag2 in tagDict:
                tranDictProb[(tag1, tag2)] = 1
                uniqueTranCount += 1
        
        # For each sentence.
        for x in range(len(sentences)):

            # For each word.
            for y in range(len(sentences[x]) - 1):
                
                # Get the first and second tags.
                tagI = sentences[x][y][1]
                tagJ = sentences[x][y+1][1]
                tranCount += 1

                # Increment the (tag1, tag2) tuple by 1 in the dictionary.
                # For each specific tagJ (second tag), increment it by 1, 
                # used for add-one smoothing.
                tranDictProb[(tagI, tagJ)] += 1
                tagDict[tagJ] += 1

        # For each transition in the transitionProb dictionary.
        for tran in tranDictProb:
            
            # Apply add-one smoothing, divide the current (count) by the
            # number of times tagJ (tran[1]) was found in sentences.
            tranDictProb[tran] = tranDictProb[tran] / (tagDict[tran[1]])

        # Set the transition probability.
        self.transition_probability = tranDictProb.copy()
        
        # Create a word dictionary.
        wordDict = {}

        # For each sentence in sentences.
        for s in sentences:

            # For each word in sentence.
            for w in s:

                # Lowercase the word.
                word = w[0].lower()

                # Add it to the dictionary if it does not exist already.
                if word not in wordDict:
                    wordDict[word] = 1

        # Create an emission probability dictionary.
        emisDictProb = {} 

        # For each tag in the tag Dictionary.
        for tag in tagDict:

            # For each word in the word dictionary, create a dictionary
            # with all possible (word, tag) combinations, set to 1 for 
            # add-one smoothing.
            for word in wordDict:
                emisDictProb[(tag, word)] = 1

        # For each sentnece in the sentences.
        for x in range(len(sentences)):

            # For eeach word in the sentence.
            for y in range(len(sentences[x])):
                
                # Get the tag and word (lowercase).
                tagI = sentences[x][y][1]
                wordJ = sentences[x][y][0].lower()

                # Increment the (tag, word) tuple found in the emisDictProb.
                # For each specific wordJ, increment it by 1, 
                # used for add-one smoothing.
                emisDictProb[(tagI, wordJ)] += 1
                wordDict[wordJ] += 1

        # For each emission in the emisDictProb dictionary.
        for emis in emisDictProb:

            # Apply add-one smoothing, divide the current (count) by the
            # number of times wordJ (emis[1]) was found in sentences.
            emisDictProb[emis] = emisDictProb[emis] / (wordDict[emis[1]])

        # Set the emission probability.
        self.emission_probability = emisDictProb.copy()
    
    # Decode the viterbi algorithm for te test sentence given.
    def viterbi_decode(self, sentence):

        # If sentence is not a string, output it.
        if type(sentence) != str:
            sys.exit("Incorrect input to method")

        # Get the lowercase of the sentence and split it the sentArray.
        sentence = sentence.lower()
        sentArray = sentence.split(' ')

        # Create a 2D viterbi array Matrix and associated Path matrix.
        viterbiMatrix = [[0 for y in range(len(tagArray))] for x in range(len(sentArray))]
        viterbiPath = [[0 for y in range(len(tagArray))] for x in range(len(sentArray))]

        # The initial tag probabilities for the sentence (the first word).
        # For each tag in the tagArray.
        for y in range(len(tagArray)):

            # Get the the relevant tag.
            tag = tagArray[y]

            try:
                # Compute the initial probability of the word and tag, add the tag to the path matrix.
                viterbiMatrix[0][y] = self.initial_tag_probability[tag] * self.emission_probability[(tag, sentArray[0])]
                viterbiPath[0][y] = tag
            
            # If the word doesn't exist in the dictionary, take an equal emission probability instead. 
            except:
                viterbiMatrix[0][y] = self.initial_tag_probability[tag] * (1 / len(tagArray))
                viterbiPath[0][y] = tag

        # For the rest of the sentence in sentArray.
        for x in range(1, len(sentArray)):
            
            # Get the word of the sentArray.
            word = sentArray[x]

            # For each tag in the tagArray.
            for y in range(len(tagArray)):

                # Set the value of tagJ.
                tagJ = tagArray[y]

                # For each tag in the tagArray.
                for z in range(len(tagArray)):

                    # Get the value of tagI.
                    tagI = tagArray[z]

                    try:
                        # If the word exists in the dictionary, get the viterbi temporary value.
                        tempValue = viterbiMatrix[x-1][z] * self.transition_probability[(tagI, tagJ)] * self.emission_probability[(tagJ, word)]
                    
                    # Otherwise, assume the tag probability is equal for the uknown word.
                    except:
                        tempValue = viterbiMatrix[x-1][z] * self.transition_probability[(tagI, tagJ)] * (1 / len(tagArray))

                    # If the tempValue is greater than the current vlaue, add it to the viterbiMatrix and the path.
                    if (tempValue > viterbiMatrix[x][y]):
                        viterbiMatrix[x][y] = tempValue
                        viterbiPath[x][y] = viterbiPath[x-1][z] + " " + tagJ 

        # Get the max probability for the must likely tagging.
        finalIndex = -1
        tempMax = -1

        # For each tag in the tagArray.
        for y in range(len(tagArray)):
            
            # If the value is greater than tempMax, have it be the tempMax.
            # Also store the index.
            if (viterbiMatrix[len(sentArray)-1][y] > tempMax):
                tempMax = viterbiMatrix[len(sentArray)-1][y]
                finalIndex = y

        # Get the path of the most likely tag probability.
        outputArray = viterbiPath[len(sentArray)-1][finalIndex].split(' ')
        
        # Return that path.
        return outputArray


# Main method (first to run)
if __name__ == "__main__":

    # Makes sure there is only 2 arguments other than "Homework1.py"
    if (len(sys.argv) != 2): 
        print("Program requires exactly 1 argument: <corpus train path>.")
        exit(-1)

    # Add the the corpus path.
    corpusPath = sys.argv[1]
    
    # Create an instance of the Tagger class.
    tagger = Tagger()
    
    # Load the path and initialize the probabilities. 
    sentenceArray = tagger.load_corpus(corpusPath)
    tagger.initialize_probabilities(sentenceArray)


    # Define the evaluation sentences.
    testString1 = "the secretariat is expected to race tomorrow ."

    testString2 = "people continue to enquire the reason for the race for outer space ."
    
    # Get the output for the evaluation sentences using viterbi decode.
    output1 = tagger.viterbi_decode(testString1)
    print(testString1)
    print(output1)
    
    output2 = tagger.viterbi_decode(testString2)
    print(testString2)
    print(output2)
