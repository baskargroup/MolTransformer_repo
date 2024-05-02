from model import *


print('imported')
MO = ModelOperator()
print('---------build model operator-----------')
#MO.evaluate_decoder(num_batch = 1)
#MO.train()
MO.r_square(num_batch = 10000)
MO.train()
MO.save_model()
MO.r_square(num_batch = 10000)
