import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import torch.fft 

class PatchEmbed(M.Model):
	def initialize(self, patch_size, stride, emb_dim):
		self.c1 = M.ConvLayer(patch_size, emb_dim, stride=stride, pad='VALID', usebias=True)
	def forward(self, x):
		x = self.c1(x)  # [B, d, H, W]
		B, d, h, w = x.shape 
		x = x.reshape(B, d, h*w).transpose(1, 2)
		return x 

class Fourier(M.Model):
	def forward(self, x):
		return torch.fft.fftn(x, dim=(-1, -2)).real 

class MLP(M.Model):
	def initialize(self, dim, mlp_ratio):
		self.fc1 = M.Dense(dim*mlp_ratio, usebias=True, activation=M.PARAM_GELU)
		self.fc2 = M.Dense(dim)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		return x 

class Block(M.Model):
	def initialize(self, dim, mlp_ratio=4, drop_path=0.0):
		self.drop_path = drop_path
		self.norm1 = M.LayerNorm(1)
		self.four = Fourier()
		self.norm2 = M.LayerNorm(1)
		self.mlp = MLP(dim, mlp_ratio)

	def forward(self, x):
		x1 = self.four(x)
		x1 = F.dropout(x1, self.drop_path, self.training, False)
		x = self.norm1(x1 + x)

		x1 = self.mlp(x)
		x1 = F.dropout(x1, self.drop_path, self.training, False)
		x = self.norm2(x1 + x)
		return x 

class PosEmbed(M.Model):
	def build_forward(self, x):
		B, N, D = x.shape 
		self.pos_emb = nn.Parameter(torch.zeros(1, N, D))
		nn.init.uniform_(self.pos_emb, -0.2, 0.2)
		return self.pos_emb + x 

	def forward(self, x):
		return self.pos_emb + x 

class FNet(M.Model):
	def initialize(self, patch_size, patch_stride, emb_dim, depth, mlp_ratio=4, drop_path=0.4):
		self.pe = PatchEmbed(patch_size, patch_stride, emb_dim)
		self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
		nn.init.uniform_(self.cls_token, -0.2, 0.2)
		self.pos_emb = PosEmbed()

		self.blocks = nn.ModuleList()
		for i in range(depth):
			# drop_rate = drop_path * i / (depth - 1)
			drop_rate = drop_path
			self.blocks.append(Block(emb_dim, mlp_ratio=mlp_ratio, drop_path=drop_rate))

	def forward(self, x):
		B = x.shape[0]
		x = self.pe(x)
		cls_token = self.cls_token.expand(B, -1, -1)
		x = torch.cat([cls_token, x], dim=1)
		
		x = self.pos_emb(x)

		for b in self.blocks:
			x = b(x)
		return x[:,0]

class FaceVit(M.Model):
	def initialize(self):
		self.trans = FNet(patch_size=16, patch_stride=8, emb_dim=512, depth=12)
		self.fc1 = M.Dense(2048, activation=M.PARAM_GELU)
		self.fc2 = M.Dense(512)

	def forward(self, x):
		x = self.trans(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return x 

if __name__=='__main__':
	net = FaceVit()

	import time 
	from tqdm import tqdm
	
	x = torch.zeros(1, 3, 112, 112)
	y = net(x)

	net.cuda()
	x = x.cuda()

	t1 = time.time()
	for i in tqdm(range(100)):
		y = net(x)
	# print(y.shape)
	t2 = time.time()
	print('Time elapsed',t2 - t1)
