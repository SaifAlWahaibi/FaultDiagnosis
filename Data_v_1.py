
# Python libraries: -

import pyreadr as pr

import numpy as np

# Seeding to ensure reproducibility: -

np.random.seed(0)

# Loading TEP data: -

training_faulty = pr.read_r(r"C:/Users/salwahai/OneDrive - Texas Tech University/LuGroup/Saif/Code/00 Datasets/01 TEP Large Size/TEP_Faulty_Training.Rdata")

testing_faulty = pr.read_r(r"C:/Users/salwahai/OneDrive - Texas Tech University/LuGroup/Saif/Code/00 Datasets/01 TEP Large Size/TEP_Faulty_Testing.Rdata")

# Extracting training data: -

training_faulty = training_faulty['faulty_training']

# Extracting testing data: -

testing_faulty = testing_faulty['faulty_testing']

# Number of simulations to extract for training, validation and testing: -

training_sim = 40

validation_sim = 5

testing_sim = 5

# Dropping non-faulty data: -

initial_training_sample = int(((1 * 60) / 3) + 1)

initial_testing_sample = int(((8 * 60) / 3) + 1)

# Choosing the simulation to include in the datasets: -

training_sim_num = np.random.choice(np.arange(1, 401), size=training_sim, replace=False)

validation_sim_num = np.random.choice(np.arange(401, 451), size=validation_sim, replace=False)

testing_sim_num = np.random.choice(np.arange(451, 501), size=testing_sim, replace=False)

# Dropping non-faulty data: -

training_faulty = training_faulty[training_faulty['simulationRun'].isin(training_sim_num)][training_faulty['sample'] >= initial_training_sample].reset_index(drop=True)

validation_faulty = testing_faulty[testing_faulty['simulationRun'].isin(validation_sim_num)][testing_faulty['sample'] >= initial_testing_sample].reset_index(drop=True)

testing_faulty = testing_faulty[testing_faulty['simulationRun'].isin(testing_sim_num)][testing_faulty['sample'] >= initial_testing_sample].reset_index(drop=True)

# Saving data: -

training_faulty.to_csv('training40t5v5t.csv', index=False)

validation_faulty.to_csv('validation40t5v5t.csv', index=False)

testing_faulty.to_csv('testing40t5v5t.csv', index=False)
