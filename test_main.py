from MolTransformer import *

'''print('imported')
MO = ModelOperator()
print('---------build model operator-----------')
MO.evaluate_decoder(num_batch = 1)
#MO.train()
MO.r_square(num_batch = 100)'''

build_model_instance = BuildModel(model_mode = 'SS')
model = build_model_instance.model
print("loaded SS model")

build_model_instance = BuildModel(dataset = 'SS')
model = build_model_instance.model
print("loaded SS model")

build_model_instance = BuildModel(dataset = 'ocelot')
model = build_model_instance.model
print("loaded multiHF_hf ocelot model")

build_model_instance = BuildModel(dataset = 'qm9')
model = build_model_instance.model
print("loaded multiHF_hf qm9 model")

data = DataLoader()
test_data = data.test

data = DataLoader(dataset = 'SS')
test_data = data.test
print("loaded DataLoader SS ")
data = DataLoader(dataset = 'ocelot')
test_data = data.test
print("loaded DataLoader ocelot ")
data = DataLoader(dataset = 'qm9')
test_data = data.test
print("loaded DataLoader qm9 ")
