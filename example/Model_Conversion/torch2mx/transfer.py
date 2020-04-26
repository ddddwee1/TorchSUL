import mxnet as mx 

prefix = './modelmx/effnet'
epoch = 0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
# all_layers = sym.get_internals()
# sym = all_layers['fc1_output']
# print(sym)
# digraph = mx.visualization.plot_network(sym, shape={'data':(1,3,112,112)}, node_attrs={"fixedsize":"false"}, hide_weights=False)
# digraph.view()

# model = mx.mod.Module(sym, data_names=['data'], label_names=None, context=mx.gpu(0))
# model.bind(data_shapes=[['data', (1,3,112,112)]])

res = {}
for k in arg_params.keys():
	dt = arg_params[k]
	if not k in res:
		res[k] = dt 
	else:
		print('Key already exsits:',k)
for k in aux_params.keys():
	dt = aux_params[k]
	if not k in res:
		res[k] = dt 
	else:
		print('Key already exsits:',k)

print('PARAMS from MX:',len(res))

# ks = list(res.keys())
# print(ks[0])
# print(res[ks[0]])

def save_model(prefix, epoch):
	model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names = None)
	model.bind(data_shapes=[('data', (1, 3, 16*4, 112))])
	model.set_params(arg_params, aux_params)
	model.save_checkpoint(prefix, epoch)

def assign_value(key, value):
	value = mx.nd.array(value)
	res[key][:] = value[:]
	print('Assigned:', key)
