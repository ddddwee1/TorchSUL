import mxnet as mx 
import numpy as np 

prefix = './r100-arcface-emoreasiaours10/model'
epoch = 62
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
sym = all_layers['relu0_output']
# sym = all_layers['fc1_output']
model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names = None)
model.bind(data_shapes=[('data', (1, 3, 112, 112))])
model.set_params(arg_params, aux_params)

input_blob = np.ones([1,3,112,112])*255
data = mx.nd.array(input_blob)
db = mx.io.DataBatch(data=(data,))
model.forward(db, is_train=False)
out = model.get_outputs()[0].asnumpy()
print(out.shape)
print(out)
