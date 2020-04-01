import mxnet as mx 

prefix = './r100-arcface-emoreasiaours10/model'
epoch = 62
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
sym = all_layers['fc1_output']
# print(sym)
digraph = mx.visualization.plot_network(sym, shape={'data':(1,3,112,112)}, node_attrs={"fixedsize":"false"}, hide_weights=False)
digraph.view()
# print(type(arg_params))
# # print(arg_params.keys())
# # print(aux_params.keys())
# print(all_layers.list_outputs())
# print(arg_params['relu0_gamma'])

# res = {}
# for k in arg_params.keys():
# 	dt = arg_params[k]
# 	dt = dt.asnumpy()
# 	if not k in res:
# 		res[k] = dt 
# 	else:
# 		print('Key already exsits:',k)
# for k in aux_params.keys():
# 	dt = aux_params[k]
# 	dt = dt.asnumpy()
# 	if not k in res:
# 		res[k] = dt 
# 	else:
# 		print('Key already exsits:',k)

# import pickle 
# pickle.dump(res ,open('params.pkl','wb'))
