import torch 
import vision_transformer 
from TorchSUL import Model as M 
import torch.nn as nn 
import config 
from vision_transformer import Block

class DepthToSpace(M.Model):
	def initialize(self, block_size):
		self.block_size = block_size
	def forward(self, x):
		bsize, chn, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		assert chn%(self.block_size**2)==0, 'DepthToSpace: Channel must be divided by square(block_size)'
		x = x.view(bsize, -1, self.block_size, self.block_size, h, w)
		x = x.permute(0,1,4,2,5,3)
		x = x.reshape(bsize, -1, h*self.block_size, w*self.block_size)
		return x 

class UpSample(M.Model):
	def initialize(self, upsample_layers, upsample_chn):
		self.prevlayers = nn.ModuleList()
		#self.uplayer = M.DeConvLayer(3, upsample_chn, stride=2, activation=M.PARAM_PRELU, batch_norm=True, usebias=False)
		self.uplayer = M.ConvLayer(3, upsample_chn*4, activation=M.PARAM_PRELU, usebias=True)
		self.d2s = DepthToSpace(2)
		self.postlayers = nn.ModuleList()
		for i in range(upsample_layers):
			self.prevlayers.append(M.ConvLayer(3, upsample_chn, activation=M.PARAM_PRELU, usebias=False, batch_norm=True))
		for i in range(upsample_layers):
			self.postlayers.append(M.ConvLayer(3, upsample_chn, activation=M.PARAM_PRELU, usebias=False, batch_norm=True))
	def forward(self, x):
		for p in self.prevlayers:
			x = p(x)
		x = self.uplayer(x)
		x = self.d2s(x)
		# print('UPUP', x.shape)
		for p in self.postlayers:
			x = p(x)
		return x 

class JointHead(M.Model):
	def initialize(self):
		self.fc = nn.Linear(384, 192)
		self.block = Block(dim=192, num_heads=3, mlp_ratio=1, qkv_bias=True, qk_scale=None,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
		self.upsample = UpSample(1, 32)
		self.conv = M.ConvLayer(1, 1)

	def forward(self, x):
		x = self.fc(x)
		x = self.block(x)
		x = x.view(-1, 28, 28, 192) # hard code here
		x = x.permute(0, 3,1,2).contiguous()
		x = self.upsample(x)
		x = self.conv(x)
		return x 

class DinoNet(M.Model):
	def initialize(self):
		self.backbone = vision_transformer.deit_small(patch_size=8)
		
		self.joints_head = nn.ModuleList()
		for i in range(config.num_pts):
			self.joints_head.append(JointHead())

	def forward(self, x):
		x = self.backbone.forward_2(x)
		res = []
		for h in self.joints_head:
			res.append(h(x))
		res = torch.cat(res, dim=1)
		return res 

def get_net():
	net = DinoNet()
	x = torch.zeros(1, 3, 224, 224)
	y = net(x)
	print(y.shape)
	checkpoint = torch.load('dino_deitsmall8_pretrain.pth', map_location='cpu')
	net.backbone.load_state_dict(checkpoint, strict=True)
	print('Network initialized')
	return net 

if __name__=='__main__':
	# net = vision_transformer.deit_small(patch_size=8)
	# checkpoint = torch.load('dino_deitsmall8_pretrain.pth', map_location='cpu')
	# net.load_state_dict(checkpoint, strict=True)

	# x = torch.zeros(1, 3, 224, 224)
	# y = net.forward_2(x)
	# print(y.shape)

	x = torch.zeros(1, 3, 224, 224)
	net = get_net()
	y = net(x)
	print(y.shape)

