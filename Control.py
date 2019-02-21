import random


class Chromosome:
	def __init__(self,filterNumbers,filterSizes):
		self.filterNumbers = filterNumbers
		self.filterSizes = filterSizes
		self.precision = -1

	def set_quality(self,quality):
		self.precision = precision


class Control:
	def __init__(self,num_layers,maxFilterChange,maxSizeChange,maxFilterNumber,maxFilterSize):
		self.num_layers = num_layers
		self.maxFilterChange = maxFilterChange
		self.maxSizeChange = maxSizeChange
		self.maxFilterNumber = maxFilterNumber
		self.maxFilterSize = maxFilterSize

	def create_chromosome(self):
		filter_size_list = []
		filter_number_list = []
		for i in range(0,self.num_layers):
			filter_size_list.append(random.randint(1,self.maxFilterSize))
			filter_number_list.append(random.randint(1,self.maxFilterNumber))
		return Chromosome(filter_number_list,filter_size_list)

	def mutate(self,chromosome):
		#We select a random index on both number and size of filter and we create a new list
		random_filter_number_index = random.randint(0, len(chromosome.filterNumbers)-1)
		random_filter_size_index = random.randint(0, len(chromosome.filterSizes)-1)
		#Create a copy of the list
		new_filterNumbers = chromosome.filterNumbers.copy()
		new_filterSizes = chromosome.filterSizes.copy()
		#get the number and size based on the index
		selected_filter_number = new_filterNumbers[random_filter_number_index]
		selected_filter_size_value = new_filterSizes[random_filter_size_index]
		#Get new values for filter and size
		random_new_filter_number = random.randint(1,self.maxFilterChange)
		random_new_size = random.randint(1,self.maxSizeChange)
		#Change the values of the chosen filter size and filter number
		if ((selected_filter_number + random_new_filter_number) >=  self.maxFilterNumber+1):
			new_filterNumbers[random_filter_number_index] = selected_filter_number - random_new_filter_number
		elif ((selected_filter_number - random_new_filter_number)  <= 0):
			new_filterNumbers[random_filter_number_index] = selected_filter_number + random_new_filter_number
		else :
			new_filterNumbers[random_filter_number_index] = selected_filter_number + random.sample(set([-(random_new_filter_number),random_new_filter_number]),1)[0]
		if ((selected_filter_size_value + random_new_size) >= self.maxFilterSize+1):
			new_filterSizes[random_filter_size_index] = selected_filter_size_value - random_new_size
		elif ((selected_filter_size_value - random_new_size) <= 0):
			new_filterSizes[random_filter_size_index] = selected_filter_size_value + random_new_size
		else:
			new_filterSizes[random_filter_size_index] = selected_filter_size_value + random.sample(set([-(random_new_size),random_new_size]),1)[0]
		return Chromosome(new_filterNumbers,new_filterSizes)

	def crossover(self,chromosome_one,chromosome_two):
		#Set random father and mother
		list = [chromosome_one,chromosome_two]
		father_chromosome = random.sample(list,1)[0]
		list.remove(father_chromosome)
		mother_chromosome = list[0]
		#Select the point of division of each list
		set = random.randint(1,self.num_layers-1)
		#Get a set from father and mother -Filters and sizes-
		new_filterNumbersFather = father_chromosome.filterNumbers[0:set].copy()
		new_filterNumbersMother = mother_chromosome.filterNumbers[set:].copy()
		new_filterSizesFather = father_chromosome.filterSizes[0:set].copy()
		new_filterSizesMother = mother_chromosome.filterSizes[set:].copy()
		#joining the sets
		new_filterNumbers = new_filterNumbersFather + new_filterNumbersMother
		new_filterSizes = new_filterSizesFather + new_filterSizesMother
		#return a new chromosome
		return Chromosome(new_filterNumbers,new_filterSizes)







########################################################################################
#############JUST FOR TEST PURPOSES######################################################
def test():
	control = Control(4,10,2,25,10)
	print('Config - |maxFilterChange {}|maxSizeChange {}|maxFilterNumber {}|maxFilterSize {}|'.format(10,2,25,10))
	list = []
	for i in range(0,5):
		list.append(control.create_chromosome())
	for r in list:
		print('------------')
		print('N' ,r.filterNumbers)
		print('S' ,r.filterSizes)
		print('------------')
	print('-----------------------MUTATTION---------------------------------------')
	for r in list:
		print('------------')
		print('N' ,r.filterNumbers)
		print('S' ,r.filterSizes)
		print('------------')
		q = control.mutate(r)
		print('-----MUTATED-------')
		print('N' ,q.filterNumbers)
		print('S' ,q.filterSizes)
		print('------------')
		print('-----NORMAL-------')
		print('N' ,r.filterNumbers)
		print('S' ,r.filterSizes)
		print('------------')
		print('--------------------------------------------------------------')
	print('-----------------------CROSSOVER---------------------------------------')
	for chromosome1,chromosome2 in zip(list[:-1],list[1:]):
		e = control.crossover(chromosome1,chromosome2)
		print('------C1------')
		print('N' ,chromosome1.filterNumbers)
		print('S' ,chromosome1.filterSizes)
		print('------------')
		print('-----C2-------')
		print('N' ,chromosome2.filterNumbers)
		print('S' ,chromosome2.filterSizes)
		print('------------')
		print('-----CROSS-------')
		print('N' ,e.filterNumbers)
		print('S' ,e.filterSizes)
		print('------------')
		print('--------------------------------------------------------------')
	print('---------------------------::::::::::------------------------------')
	for r in list:
		print('------------')
		print('N' ,r.filterNumbers)
		print('S' ,r.filterSizes)
		print('------------')

#if __name__ == "__main__":
#	test()
