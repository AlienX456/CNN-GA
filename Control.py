import random


class Chromosome:
	def __init__(self, filterNumbers, filterSizes, maxFilterSizePerLayer):
		self.filterNumbers = filterNumbers
		self.filterSizes = filterSizes
		self.maxFilterSizePerLayer = maxFilterSizePerLayer
		self.precision = -1
		self.model = None
		self.history = None

	def set_quality(self, precision):
		self.precision = precision


# MAXFILTERSIZE will be the width/height of the image.. with take every time a maxpooling with stride 2x2
class Control:
	def __init__(self, num_layers, maxFilterChange, maxSizeChange, maxFilterNumber, maxFilterSize, img_size):
		self.num_layers = num_layers
		self.maxFilterChange = maxFilterChange
		self.maxSizeChange = maxSizeChange
		self.maxFilterNumber = maxFilterNumber
		self.maxFilterSize = maxFilterSize
		self.img_size = img_size

	def validate_chromosome_max_sizes(self, sizes_list):
		current_output = self.img_size
		max_sizes_list = []
		for i in range(0, self.num_layers):
			if (sizes_list[i] > current_output):
				sizes_list[i] = current_output
			elif (sizes_list[i] <= 0):
				sizes_list[i] = 1
			max_sizes_list.append(current_output)
			if i < self.num_layers - 1:
				current_output = int((current_output - sizes_list[i] + 1) / 2)
			else:
				current_output = int(current_output - sizes_list[i] + 1)

			if current_output <= 0:
				return None

		return max_sizes_list

	def create_chromosome(self):
		filter_size_list = []
		filter_number_list = []
		max_size_list_per_layer = []
		current_output = self.img_size
		for i in range(0, self.num_layers):
			random_filter_size = random.randint(1, self.maxFilterSize)
			filter_size_list.append(random_filter_size)
			max_size_list_per_layer.append(current_output)
			# This is because the last layer has not pooling
			if i < self.num_layers - 1:
				current_output = int((current_output - filter_size_list[-1] + 1) / 2)
			else:
				current_output = int(current_output - filter_size_list[-1] + 1)
			# If the current chromosome doesn't accomplish the output condition, we use recursion to fix it
			if current_output <= 0:
				return self.create_chromosome()
			filter_number_list.append(random.randint(1, self.maxFilterNumber))
		return Chromosome(filter_number_list, filter_size_list, max_size_list_per_layer)

	def mutate(self, chromosome):
		# We select a random index on both number and size of filter and we create a new list
		random_filter_number_index = random.randint(0, len(chromosome.filterNumbers) - 1)
		random_filter_size_index = random.randint(0, len(chromosome.filterSizes) - 1)
		# Create a copy of the list
		new_filterNumbers = chromosome.filterNumbers.copy()
		new_filterSizes = chromosome.filterSizes.copy()
		# get the number and size based on the index
		selected_filter_number = new_filterNumbers[random_filter_number_index]
		selected_filter_size_value = new_filterSizes[random_filter_size_index]

		# Get new values for filter and size
		random_new_filter_number = random.randint(1, self.maxFilterChange)
		random_new_size = random.randint(1, self.maxSizeChange)

		# Change the values of the chosen filter number
		new_filter_number = selected_filter_number + \
							random.sample(set([-(random_new_filter_number), random_new_filter_number]), 1)[0]
		if new_filter_number >= (self.maxFilterNumber + 1):
			new_filterNumbers[random_filter_number_index] = self.maxFilterNumber
		elif new_filter_number <= 0:
			new_filterNumbers[random_filter_number_index] = 1
		else:
			new_filterNumbers[random_filter_number_index] = new_filter_number

		# Change the values of the chosen filter size
		new_size_value = selected_filter_size_value + random.sample(set([-(random_new_size), random_new_size]), 1)[0]
		if new_size_value >= (chromosome.maxFilterSizePerLayer[random_filter_size_index] + 1):
			new_filterSizes[random_filter_size_index] = chromosome.maxFilterSizePerLayer[random_filter_size_index]
		elif new_size_value <= 0:
			new_filterSizes[random_filter_size_index] = 1
		else:
			new_filterSizes[random_filter_size_index] = new_size_value

		# Once mutated a gen, we must verify again if al the filter sizes are good
		max_sizes_list = self.validate_chromosome_max_sizes(new_filterSizes)

		if max_sizes_list is not None:
			return Chromosome(new_filterNumbers, new_filterSizes, max_sizes_list)
		else:
			return self.mutate(chromosome)

	def crossover(self, chromosome_one, chromosome_two):
		# Set random father and mother
		list = [chromosome_one, chromosome_two]
		father_chromosome = random.sample(list, 1)[0]
		list.remove(father_chromosome)
		mother_chromosome = list[0]
		# Select the point of division of each list
		set = random.randint(1, self.num_layers - 1)
		# Get a set from father and mother -Filters and sizes-
		new_filterNumbersFather = father_chromosome.filterNumbers[0:set].copy()
		new_filterNumbersMother = mother_chromosome.filterNumbers[set:].copy()
		new_filterSizesFather = father_chromosome.filterSizes[0:set].copy()
		new_filterSizesMother = mother_chromosome.filterSizes[set:].copy()
		# joining the sets
		new_filterNumbers = new_filterNumbersFather + new_filterNumbersMother
		new_filterSizes = new_filterSizesFather + new_filterSizesMother
		max_filter_sizes = self.validate_chromosome_max_sizes(new_filterSizes)
		# return a new chromosome
		if max_filter_sizes is not None:
			return Chromosome(new_filterNumbers, new_filterSizes, max_filter_sizes)
		else:
			return self.crossover(chromosome_one, chromosome_two)


########################################################################################
#############JUST FOR TEXT PURPOSES######################################################
def test():
	# num_layers, maxFilterChange, maxSizeChange, maxFilterNumber, maxFilterSize, img_size
	control = Control(4, 10, 2, 25, 10, 28)
	print('Config - |maxFilterChange {}|maxSizeChange {}|maxFilterNumber {}|maxFilterSize {}|'.format(10, 2, 25, 10))
	list = []
	for i in range(0, 5):
		list.append(control.create_chromosome())
	for r in list:
		print('------------')
		print('N-Filters: ', r.filterNumbers)
		print('S-Filters: ', r.filterSizes)
		print('Last Output: ', r.maxFilterSizePerLayer[-1])
		print('------------')
	print('-----------------------MUTATTION---------------------------------------')
	for r in list:
		print('------------')
		print('N-Filters: ', r.filterNumbers)
		print('S-Filters: ', r.filterSizes)
		print('Last Output: ', r.maxFilterSizePerLayer[-1])
		print('------------')
		q = control.mutate(r)
		print('-----MUTATED-------')
		print('N-Filters: ', q.filterNumbers)
		print('S-Filters: ', q.filterSizes)
		print('Last Output: ', r.maxFilterSizePerLayer[-1])
		print('------------')
		print('--------------------------------------------------------------')
	print('-----------------------CROSSOVER---------------------------------------')
	for chromosome1, chromosome2 in zip(list[:-1], list[1:]):
		e = control.crossover(chromosome1, chromosome2)
		print('------C1------')
		print('N', chromosome1.filterNumbers)
		print('S', chromosome1.filterSizes)
		print('------------')
		print('-----C2-------')
		print('N', chromosome2.filterNumbers)
		print('S', chromosome2.filterSizes)
		print('------------')
		print('-----CROSS-------')
		print('N-Filters: ', e.filterNumbers)
		print('S-Filters: ', e.filterSizes)
		print('Last Output: ', e.maxFilterSizePerLayer[-1])
		print('------------')
		print('--------------------------------------------------------------')
	print('---------------------------::::::::::------------------------------')
	for r in list:
		print('------------')
		print('N', r.filterNumbers)
		print('S', r.filterSizes)
		print('------------')


if __name__ == "__main__":
	test()
