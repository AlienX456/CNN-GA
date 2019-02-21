from Simulator import Simulator
from Control import Control

def main():
	individuals = 3
	survivors = 5
	control = Control(4,10,2,25,10)
	#control = Control(num_layers,maxFilterChange,maxSizeChange,maxFilterNumber,maxFilterSize)
	simulator = Simulator(individuals,control,survivors,100,None)
	#self,initialPoblation,control,num_survivors,condition_precision,condition_generations
	simulator.simulate()

if __name__ == "__main__":
	main()
