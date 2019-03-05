from Simulator import Simulator
from Control import Control
from CNN import CNN

def main():
	img_size = 28
	num_layers = 3
	batch_size = 256
	num_classes  = 10
	num_epoch = 7
	#(self, num_conv_layers, data_train_path, data_test_path, img_rows, img_cols, epoch_number, batch_size, num_classes)
	cnn = CNN(num_layers, 'data/fashionmnist/fashion-mnist_train.csv', 'data/fashionmnist/fashion-mnist_test.csv', img_size, img_size, num_epoch, batch_size, num_classes)

	prom0 = 0
	for i in range (0,3):
		cnn.configure_data()
		result = cnn.training_cnn([8,8,8],[5,2,1])
		prom0 = prom0 + (result[2][1])/3


	prom1 = 0
	for i in range (0,3):
		cnn.configure_data()
		result = cnn.training_cnn([10,10,10],[3,2,2])
		prom1 = prom1 + (result[2][1])/3


	prom2 = 0
	for i in range (0,3):
		cnn.configure_data()
		result = cnn.training_cnn([9,10,9],[6,3,1])
		prom2 = prom2 + (result[2][1])/3


	prom3 = 0
	for i in range (0,3):
		cnn.configure_data()
		result = cnn.training_cnn([11,12,14],[4,3,1])
		prom3 = prom3 + (result[2][1])/3


	prom4 = 0
	for i in range (0,3):
		cnn.configure_data()
		result = cnn.training_cnn([8,9,8],[2,1,1])
		prom4 = prom4 + (result[2][1])/3

	print("\n\n\n\n\n")
	print("Prom [5,5,5],[10,3,2] : ",prom0)
	print("Prom [5,5,5],[5,2,1]: ",prom1)
	print("Prom [8,8,8],[10,3,2] : ",prom2)
	print("Prom [2,2,2],[10,3,2] : ",prom3)
	print("Prom [8,8,8],[5,2,1] : ",prom4)



if __name__ == "__main__":
	main()
