import mxnet as mx 
import numpy as np 

prefix = './mx/mnet.25'
epoch = 0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

# digraph = mx.visualization.plot_network(sym, shape={'data':(1,3,640,640)}, node_attrs={"fixedsize":"false"}, hide_weights=False)
# digraph.view()

all_layers = sym.get_internals()
# lys = all_layers.list_outputs()
# for l in lys:
# 	print(l)
sym = all_layers['face_rpn_bbox_pred_stride8_output']
model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names = None)
model.bind(data_shapes=[('data', (1, 3, 640, 640))])
model.set_params(arg_params, aux_params)

input_blob = np.ones([1,3,640,640])
data = mx.nd.array(input_blob)
db = mx.io.DataBatch(data=(data,))
model.forward(db, is_train=False)
out = model.get_outputs()[0].asnumpy()
print(out)
print(out.shape)

# res = {}
# for k in arg_params.keys():
# 	dt = arg_params[k]
# 	if not k in res:
# 		res[k] = dt 
# 	else:
# 		print('Key already exsits:',k)
# for k in aux_params.keys():
# 	dt = aux_params[k]
# 	if not k in res:
# 		res[k] = dt 
# 	else:
# 		print('Key already exsits:',k)
