import numpy as np
import math

#Global variable as these values never change
pos_tags = {'A':0, 'C':1, 'D':2, 'M':3, 'N':4, 'O':5, 'P':6, 'R':7, 'V':8, 'W':9}
low_threshold = 2
alpha = 1e-50
beta = 1e-50

def load_data(filename):
	f = open(filename)
	data = f.read()
	data = data.split('\n')
	data = data[:-1]
	
	word_freq_dict = {}
	low_freq_words = {}

	for line in data:
		word_tags = line.split(' ')
		for i in word_tags:
			word,tag = i.split('/')[0],i.split('/')[1]
			word = word.lower()
			if word not in word_freq_dict:
				word_freq_dict[word] = 1
			else:
				word_freq_dict[word]+=1

	

	low_freq_words = map_low_freq_words(word_freq_dict,low_freq_words)

	print(len(low_freq_words))
	print(len(word_freq_dict))
	return word_freq_dict,low_freq_words

def map_low_freq_words(word_freq_dict,low_freq_words):
	for i in word_freq_dict:
		if word_freq_dict[i] <=low_threshold:
			low_freq_words[i] = "UNK"
	return low_freq_words

def check_low_freq(word,low_freq_words,word_freq_dict):
	try:
		return low_freq_words[word]
	except:
		if word not in word_freq_dict:
			return "UNK"
		else:
			return word

def calculate_prob_em(emission_probability_table):
	sums = {}
	for tag in range(0,len(pos_tags)):
		s = 0
		for i in emission_probability_table:
			s+=emission_probability_table[i][tag]
		sums[tag] = s

	for tag in range(0,len(pos_tags)):
		for i in emission_probability_table:
			emission_probability_table[i][tag]= (emission_probability_table[i][tag]+beta)/(sums[tag]+beta*len(emission_probability_table))

	return emission_probability_table


def calc_emission_prob(filename,word_freq_dict,low_freq_words):
	emission_probability_table = {}

	f = open(filename)
	data = f.read()
	data = data.split('\n')
	data = data[:-1]

	for line in data:
		word_tags = line.split(' ')
		for i in word_tags:
			word,tag = i.split('/')[0],i.split('/')[1]
			word = word.lower()
			
			word = check_low_freq(word,low_freq_words,word_freq_dict)

			if word not in emission_probability_table:
				emission_probability_table[word] = [0 for i in range(len(pos_tags))]

			emission_probability_table[word][pos_tags[tag]] +=1



	emission_probability_table = calculate_prob_em(emission_probability_table)

	# for i in emission_probability_table:
	# 	print(i,emission_probability_table[i])
	# print(emission_probability_table["UNK"])

	return emission_probability_table

def calculate_prob_tr(transmission_probability_table):
	for i in range(0,len(transmission_probability_table)):
		s = sum(transmission_probability_table[i])
		for j in range(0,len(transmission_probability_table[i])):
			transmission_probability_table[i][j]=(transmission_probability_table[i][j]+alpha)/(s+alpha*(len(transmission_probability_table)+1))

	return transmission_probability_table

def calc_transmission_prob(filename):
	transmission_probability_table = []
	rows=pos_tags.copy()
	cols=pos_tags.copy()

	rows['START'] = 10
	cols['END'] = 10

	print(rows)
	print(cols)

	for i in range(len(rows)):
		transmission_probability_table.append([0 for i in range(0,len(cols))])


	f = open(filename)
	data = f.read()
	data = data.split('\n')
	data = data[:-1]

	for line in data:
		previous = None
		word_tags = line.split(' ')
		for i in word_tags:
			_,tag = i.split('/')[0],i.split('/')[1]
			if previous==None:
				transmission_probability_table[rows['START']][cols[tag]]+=1
			else:
				transmission_probability_table[rows[previous]][cols[tag]]+=1

			previous = tag
		transmission_probability_table[rows[tag]][cols['END']]+=1

	# print(transmission_probability_table)
	transmission_probability_table = calculate_prob_tr(transmission_probability_table)
	# print(transmission_probability_table)
	return transmission_probability_table,rows,cols

def convert2log(emission_probability_table,transmission_probability_table):
	for i in emission_probability_table:
		for j in range(len(emission_probability_table[i])):
			emission_probability_table[i][j] = math.log(emission_probability_table[i][j],2)

	for i in range(len(transmission_probability_table)):
		for j in range(len(transmission_probability_table[i])):
			transmission_probability_table[i][j] = math.log(transmission_probability_table[i][j],2)

	return emission_probability_table,transmission_probability_table


def viterbi(emission_probability_table,transmission_probability_table,rows,cols,word_freq_dict,low_freq_words,filename):
	f = open(filename)
	data = f.read()
	data = data.split('\n')
	data = data[:-1]

	acc=0
	tot=0

	for line in data:
		word_tags = line.split(' ')
		ground_truth_tags = []

		dynamic_prob_matrix = [[0 for j in range(len(word_tags))] for i in range(len(pos_tags))]
		predicted_tag=[]
		# predicted_tag.append('START')

		word,tag = word_tags[0].split('/')[0],word_tags[0].split('/')[1]
		word=word.lower()
		word = check_low_freq(word,low_freq_words,word_freq_dict)
		for pos_tag in range(0,len(pos_tags)):
			dynamic_prob_matrix[pos_tag][0]
			transmission_probability_table[rows['START']][pos_tag]
			emission_probability_table[word][pos_tag]
			dynamic_prob_matrix[pos_tag][0] = transmission_probability_table[rows['START']][pos_tag]*emission_probability_table[word][pos_tag] 
		
		for index in range(1,len(word_tags)):
			word,tag = word_tags[index].split('/')[0],word_tags[index].split('/')[1]
			ground_truth_tags.append(tag)
			word = word.lower()
			word = check_low_freq(word,low_freq_words,word_freq_dict)
			
			max_final=[]
			for pos_tag_current in range(0,len(pos_tags)):
				max_transition_emission=[]
				for pos_tag_previous in range(0,len(pos_tags)):
					s = dynamic_prob_matrix[pos_tag_previous][index-1]*transmission_probability_table[pos_tag_previous][pos_tag_current]*emission_probability_table[word][pos_tag_current]
					max_transition_emission.append((s,pos_tag_previous))
				max_transition_emission = sorted(max_transition_emission)
				dynamic_prob_matrix[pos_tag_current][index] = max_transition_emission[0][0]
				max_final.append((dynamic_prob_matrix[pos_tag_current][index],pos_tag_current))
			
		
			max_final=sorted(max_final)
			for t,n  in pos_tags.items():
				if n == max_final[0][1]:
					predicted_tag.append(t)
					break

		#This is kind of useless, just used to assert the correct path and for completeness
		final_transitions=[]
		for pos_tag_previous in range(0,len(pos_tags)):
			final_transitions.append(dynamic_prob_matrix[pos_tag_previous][len(word_tags)-1]*transmission_probability_table[pos_tag_previous][cols['END']])

		for i in range(len(ground_truth_tags)):
			if ground_truth_tags[i]==predicted_tag[i]:
				acc+=1
			tot+=1

		# print(dynamic_prob_matrix)
		# exit(0)
	print(acc/tot)



def main():
	word_freq_dict,low_freq_words = load_data("./data/trn.pos")
	emission_probability_table = calc_emission_prob("./data/trn.pos",word_freq_dict,low_freq_words)
	transmission_probability_table,rows,cols = calc_transmission_prob("./data/trn.pos")
	emission_probability_table,transmission_probability_table = convert2log(emission_probability_table,transmission_probability_table)

	viterbi(emission_probability_table,transmission_probability_table,rows,cols,word_freq_dict,low_freq_words,"./data/tst.pos")





main()