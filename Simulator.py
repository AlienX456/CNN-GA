from Control import Control
import random
import sys

class Simulator:
    def __init__(self, initialPoblation, control, num_survivors, condition_precision, condition_generations, cnn, sim_per_config):
        self.individuals = []
        self.initialPoblation = initialPoblation
        self.control = control
        self.condition_precision = condition_precision
        self.condition_generations = condition_generations
        self.num_survivors = num_survivors
        self.cnn = cnn
        self.current_best_model = None
        self.sim_per_config = sim_per_config

    def simulate(self):
        print('Welcome lets start the Simulation >:v ')
        print('Our current config is : |num_layers {}|maxFilterChange {}|'.format(self.control.num_layers,self.control.maxFilterChange))
        print('|maxSizeChange {}|maxFilterNumber {}|maxFilterSize {}|'.format(self.control.maxSizeChange,self.control.maxFilterNumber,self.control.maxFilterSize))

        #Create initial poblation
        for i in range(0,self.initialPoblation):
            self.individuals.append(self.control.create_chromosome())
        print('Initial poblation created')

        #We start from generation 1
        current_generation = 0

        #We create a condition to control the while loop
        condition = True

        while(condition):
            #We select the first survivors that after the first iteration are already ordered
            #on first iteration is not necesary
            current_generation += 1
            print('\n','\n','\n','\n')
            print('GENERATION {} ----------------------------------------------'.format(current_generation))
            for individual in self.individuals[self.num_survivors:]:
                self.individuals.remove(individual)
            #Note: verify whether the last operation deletes data or it's still on memory
            self.print_list_chromosomes(self.individuals)

            #We start the mutation
            print('-----------------------MUTATTION---------------------------------------')
            mutated = []
            for chromosome in self.individuals:
                print('------------')
                print('Filter Numbers' ,chromosome.filterNumbers)
                print('Filter Sizes' ,chromosome.filterSizes)
                print('Precision' ,chromosome.precision)
                print('------------')
                mutated.append(self.control.mutate(chromosome))
                print('-----MUTATED-------')
                print('Filter Numbers', mutated[-1].filterNumbers)
                print('Filter Sizes' , mutated[-1].filterSizes)
                print('------------')
            #adding mutated chromosomes to individuals list
            #Important add both mutated and crossover after crossover or inmmediately

            print('\n','\n')

            #We start the crossover operation
            crossovered = []
            print('-----------------------CROSSOVER---------------------------------------')
            for chromosome_one,chromosome_two in zip(self.individuals[:-1],self.individuals[1:]):
                crossovered.append(self.control.crossover(chromosome_one,chromosome_two))
                print('------C1------')
                print('Filter Numbers' ,chromosome_one.filterNumbers)
                print('Filter Sizes'  ,chromosome_one.filterSizes)
                print('Precision' ,chromosome_one.precision)
                print('------------')
                print('------C2-------')
                print('Filter Numbers' ,chromosome_two.filterNumbers)
                print('Filter Sizes'  ,chromosome_two.filterSizes)
                print('Precision' ,chromosome_two.precision)
                print('------------')
                print('-----CROSS-------')
                print('Filter Numbers' ,crossovered[-1].filterNumbers)
                print('Filter Sizes'  ,crossovered[-1].filterSizes)
                print('------------')
            #Adding crossovered chromosomes to individuals


            self.individuals = self.individuals + mutated

            self.individuals = self.individuals + crossovered

            print('\n','\n')

            #We start the evaluation (For now using a test function)
            for chromosome in self.individuals:
                if chromosome.precision == -1:
                    chromosome.precision = 0
                    print('▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿▿')
                    print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
                    print('Starting Evaluation of :')
                    print('Filter Numbers', chromosome.filterNumbers)
                    print('Filter Sizes',chromosome.filterSizes)
                    model_history_score_list=[]
                    for i in range(0,self.sim_per_config):
                        print('---Starting ',i+1,' Evaluation of the model')
                        model_history_score_list.append(self.cnn.training_cnn(chromosome.filterNumbers, chromosome.filterSizes))
                        chromosome.precision = chromosome.precision + model_history_score_list[-1][2][1]/self.sim_per_config
                    model_history_score_list.sort(key=lambda x:x[2][1],reverse=True)
                    chromosome.model = model_history_score_list[0][0]
                    chromosome.history = model_history_score_list[0][1]
                    print('**Mean = ',chromosome.precision)
                    print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
                    print('▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵▵')
            print('--Evaluation of chromosomes completed--')

            print('\n','\n')

            #We order the evaluated chromosomes
            self.individuals.sort(key=lambda x:x.precision,reverse=True)
            print('--Sorting of chromosomes completed--')

            print('\n','\n')

            #Show the current individuals of the generation
            print('------Generation {} individuals [Result]------'.format(current_generation))
            self.print_list_chromosomes(self.individuals)

            #Keep the best model of each generation

            #verify the condition to stop
            if self.condition_precision is not None:
                if self.individuals[0].precision >= self.condition_precision:
                    condition = False
            elif self.condition_generations is not None:
                if current_generation == self.condition_generations:
                    condition = False

        #after loop we take just survivors and we generate a summary
        for individual in self.individuals[1:]:
            self.individuals.remove(individual)
        print('\n','\n','\n','\n')
        print('SUMMARY---------------')
        print('Last Generation = {}'.format(current_generation))
        print('Best Chromosome/Configuration = ')
        print('--Filter Numbers' ,self.individuals[0].filterNumbers)
        print('--Filter Sizes'  ,self.individuals[0].filterSizes)
        print('--Precision' ,self.individuals[0].precision)
        print('---------------')
        self.cnn.generate_classification_report(self.individuals[0].model)
        self.cnn.generate_precision_graph(self.individuals[0].history)
        print('\n','\n','\n','\n')
        print('We know there are other Bioinspired algorithms, but thanks for chosing this One :v')
        print('Esteban Romero 20151020048')
        print('Diego -- 20151020048')

        sys.exit(0)

    #test function to test the evaluation of chromosomes
    def test_evaluate_chromosome(self,chromosome):
        if chromosome.precision == -1 :
            chromosome.precision = random.randint(0,100)

    def print_list_chromosomes(self,list):
        for chromosome in list:
            print('------------')
            print('Filter Numbers' ,chromosome.filterNumbers)
            print('Filter Sizes' ,chromosome.filterSizes)
            print('Precision' ,chromosome.precision)
            print('------------')
