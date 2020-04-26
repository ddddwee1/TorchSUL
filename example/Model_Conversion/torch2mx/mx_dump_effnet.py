import mxnet as mx

def MBConvBlock(x, ksize, output_filters, stride, expand_ratio, name, sc=False):
	filters_ = expand_ratio * output_filters
	inp = x 

	x = mx.sym.BatchNorm(data = x, fix_gamma=False, eps=2e-5, momentum=0.9, name=name+'_prebn')

	x = mx.sym.Convolution(data = x, num_filter=filters_, kernel=(1,1), stride=(1,1),
							pad=(0,0), no_bias=True, name=name+'_conv0')
	x = mx.sym.BatchNorm(data = x, fix_gamma=False, eps=2e-5, momentum=0.9, name=name+'_bn0')
	x = mx.sym.LeakyReLU(data = x, act_type='prelu', name = name+'_relu0')
	x = mx.sym.Convolution(data=x,num_filter=filters_, kernel=(ksize, ksize), stride=(stride, stride), 
							num_group=filters_, pad=(ksize//2, ksize//2), no_bias=True, name=name+'_conv1')
	x = mx.sym.BatchNorm(data = x, fix_gamma=False, eps=2e-5, momentum=0.9, name=name+'_bn1')
	x = mx.sym.LeakyReLU(data = x, act_type='prelu', name=name+'_relu1')

	# se 
	se = mx.sym.Pooling(data=x, global_pool=True, kernel=(7,7), pool_type='avg', name=name+'_sepool1')
	se = mx.sym.Convolution(data=se, num_filter=filters_//8, kernel=(1,1), stride=(1,1), pad=(0,0),
							no_bias=False, name=name+'_seconv2')
	se = mx.sym.LeakyReLU(data = se, act_type='prelu', name=name+'_serelu2')
	se = mx.sym.Convolution(data = se, num_filter=filters_, kernel=(1, 1), stride=(1, 1), 
							pad=(0,0), no_bias=False, name=name+'_seconv3')
	se = mx.sym.Activation(data = se, act_type='sigmoid', name=name+'_sigmoid3')
	x = mx.sym.broadcast_mul(x, se)

	x = mx.sym.Convolution(data=x, num_filter=output_filters, kernel=(1,1), stride=(1,1), pad=(0,0),
							no_bias=True, name=name+'_conv4')
	x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9, name=name+'_bn4')

	if sc:
		inp = mx.sym.Convolution(data=inp, num_filter=output_filters, kernel=(1,1), stride=(stride, stride), pad=(0,0),
							no_bias=True, name=name+'_convsc')
		inp = mx.sym.BatchNorm(data=inp, fix_gamma=False, eps=2e-5, momentum=0.9, name=name+'_bnsc')

	x = inp + x 
	return x 

def Phi(x, outchn):
	buff = []
	for i in range(len(x)):
		print(i)
		if i>0:
			if i==1:
				filt = 192
			else:
				filt = 320
			b = mx.sym.pad(x[i], mode='edge', pad_width=(0,0,0,0,1,1,1,1))
			# b = mx.sym.UpSampling(data=b, num_filter=filt, scale=2, sample_type='bilinear', name="upsampling%d"%i)
			b = mx.sym.Deconvolution(data=b, num_filter=filt, kernel=(4,4), num_group=filt, no_bias=True, stride=(2,2), name='upsampling%d'%i)
			b = mx.sym.crop(data=b, begin=(None, None, 3, 3), end=(None, None, -3, -3))
			# print(b)
		else:
			b = x[i]
		buff.append(b)
	x = mx.sym.Concat(*buff, dim=1, name='phi_cat')
	x = mx.sym.Convolution(data = x, num_filter=outchn, kernel=(3,3), stride=(1,1),
						pad=(1,1), no_bias=False, name='phi_conv0')
	return x 

def EffNet(ksizes, channels, strides, expansions, repeats, finalType='E'):
	x = mx.sym.Variable(name='data')
	x = mx.sym.Convolution(data=x,num_filter=32, kernel=(3, 3), stride=(1, 1), 
							pad=(1,1), no_bias=True, name='c1')
	x = mx.sym.BatchNorm(data = x, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn1')
	x = mx.sym.LeakyReLU(data = x, act_type='prelu', name='relu1')

	feats = []
	for i, (k,c,s,e,r) in enumerate(zip(ksizes, channels, strides, expansions, repeats)):
		for j in range(r):
			sc = False
			if i==0 and j==0:
				sc = True
			if j==0:
				sc = True 

			x = MBConvBlock(x, k, c, s if j==0 else 1, e, name='Stage%d_Unit%d'%(i,j), sc=sc)
		feats.append(x)
		print(x.name)

	x = Phi(feats[-3:], 512)
	return x 

# repeats = [1,2,2,3,3,4,1]
# channels = [16, 24, 40, 80, 112, 192, 320]
# ksizes = [3,3,5,3,5,5,3]
# strides = [1,2,2,1,2,2,1]
# expansions = [1,6,6,6,6,6,6]

repeats = [1,2,2,4,3,4,3]
channels = [16, 24, 40, 80, 112, 192, 320]
ksizes = [3,3,5,3,5,5,3]
strides = [1,2,2,1,2,2,1]
expansions = [1,6,6,6,6,6,6]

effnet = EffNet(ksizes, channels, strides, expansions, repeats, 'E')
model = mx.mod.Module(context=mx.cpu(), symbol=effnet, label_names=())
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)
model.bind(data_shapes=[('data', (1, 3, 112, 112))])
model.init_params(initializer)

import os 
if not os.path.exists('./modelmx/'):
	os.mkdir('./modelmx/')
model.save_checkpoint('./modelmx/effnet', 0)
