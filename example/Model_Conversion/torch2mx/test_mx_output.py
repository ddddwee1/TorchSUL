import mxnet as mx 
import numpy as np 

prefix = './transferred/efflargeE'
# prefix = './modelmx/effnet'
epoch = 0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
# sym = all_layers['upsampling1_output']
sym = all_layers['phi_conv0_output']
# sym = all_layers['_plus15_output']
model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names = None)
model.bind(data_shapes=[('data', (1, 3, 112, 112))])
model.set_params(arg_params, aux_params)
input_blob = np.zeros([1,3,112,112])
data = mx.nd.array(input_blob)
db = mx.io.DataBatch(data=(data,))
model.forward(db, is_train=False)

out = model.get_outputs()[0].asnumpy()
# print(out)
# out = model.get_outputs()[0].asnumpy()
# print('asdf')
# print(out.shape)
# print(out)

### sorting and pooling 
# N,c,h,w = out.shape
# out = out.reshape([N,c,h*w])
# out = np.sort(out, axis=-1)
# keep = int(h*w*0.9)
# out = out[:,:,-keep:]
# feature = np.mean(out, axis=-1)
# feature = feature / np.linalg.norm(feature, axis=-1, keepdims=True)
# print(feature)
print(out)
print(out.shape)
