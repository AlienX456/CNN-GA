from Simulator import Simulator
from Control import Control
from CNN import CNN

def main():
	individuals = 3
	survivors = 3
	img_size = 28
	num_layers = 3
	batch_size = 256
	num_classes  = 10
	num_epoch = 7
	max_number_filter_change = 4
	max_size_filter_change = 1
	max_number_filter = 9
	max_filter_size = 2
	#(self, num_layers, maxFilterChange, maxSizeChange, maxFilterNumber, maxFilterSize, img_size)
	control = Control(num_layers, max_number_filter_change, max_size_filter_change, max_number_filter, max_filter_size , img_size)
	#(self, num_conv_layers, data_train_path, data_test_path, img_rows, img_cols, epoch_number, batch_size, num_classes)
	cnn = CNN(num_layers, 'data/fashionmnist/fashion-mnist_train.csv', 'data/fashionmnist/fashion-mnist_test.csv', img_size, img_size, num_epoch, batch_size, num_classes)
	cnn.configure_data()
	#(self, initialPoblation, control, num_survivors, condition_precision, condition_generations, cnn, sim_per_config)
	simulator = Simulator(individuals, control, survivors, 0.88, None, cnn, 3)
	simulator.simulate()

if __name__ == "__main__":
	main()




#PROBED CONFIGURATION 1
	# individuals = 3
	# survivors = 3
	# img_size = 28
	# num_layers = 3
	# batch_size = 256
	# num_classes  = 10
	# num_epoch = 3
	# max_number_filter_change = 10
	# max_size_filter_change = 2
	# max_number_filter = 25
	# max_filter_size = 10


#
