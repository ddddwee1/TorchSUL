import TorchSUL.Model as M 
import numpy as np 
import torch 
import torch.nn as nn 
from torch.nn.parameter import Parameter
import torch.nn.init as init 
import torch.nn.functional as F 

class MultiHeadAtt(M.Model):
	def initialize(self, num_heads, dim, drop):
		self.dim = dim 
		self.num_heads = num_heads
		self.drop = drop 

	def build(self, *inputs):
		indimq = inputs[0].shape[2] # q: [N, seq_len, dim]
		indimk = inputs[0].shape[2] # k: [N, seq_len, dim]
		indimv = inputs[0].shape[2] # v: [N, seq_len, dim]
		self.wq = Parameter(torch.Tensor(self.num_heads, self.dim, indimq))
		self.wk = Parameter(torch.Tensor(self.num_heads, self.dim, indimk))
		self.wv = Parameter(torch.Tensor(self.num_heads, self.dim, indimv))
		self.bq = Parameter(torch.Tensor(self.num_heads, self.dim))
		self.bk = Parameter(torch.Tensor(self.num_heads, self.dim))
		self.bv = Parameter(torch.Tensor(self.num_heads, self.dim))

		init.normal_(self.wq, std=0.001)
		init.normal_(self.wk, std=0.001)
		init.normal_(self.wv, std=0.001)
		init.zeros_(self.bq)
		init.zeros_(self.bk)
		init.zeros_(self.bv)

	def forward(self, q, k, v):
		# qkv: [N, seq_len, indim]
		seq_len = q.shape[1]
		q = torch.einsum('ijk,lmk->ijlm', q, self.wq) + self.bq   # q: [N, seq_len, num_heads, dim]
		k = torch.einsum('ijk,lmk->ijlm', k, self.wk) + self.bk   # k: [N, seq_len, num_heads, dim]
		v = torch.einsum('ijk,lmk->ijlm', v, self.wv) + self.bv   # v: [N, seq_len, num_heads, dim]
		
		qk = torch.einsum('ijkl,imkl->ijkm', q, k)
		qk = torch.softmax(qk / np.sqrt(self.dim), dim=-1)  # qk: [N, seq_len_q, num_heads, seq_len_k]
		if self.drop>0:
			qk = F.dropout(qk, self.drop, self.training, False)

		res = torch.einsum('ijkl,ilkm->ijkm', qk, v)
		res = res.reshape(-1, seq_len, self.dim * self.num_heads)
		return res 

class Transformer(M.Model):
	def initialize(self, num_heads, dim_per_head, drop):
		self.att = MultiHeadAtt(num_heads, dim_per_head, drop)
		self.l1 = M.Dense(dim_per_head * num_heads * 4)
		self.l2 = M.Dense(dim_per_head * num_heads)
		self.ln1 = M.LayerNorm(1)
		self.ln2 = M.LayerNorm(1)
		self.drop = drop

	def forward(self, x):
		sc = x 
		x = self.ln1(x)
		# print(x.shape)
		x = self.att(x, x, x)
		x = F.dropout(x, self.drop, self.training, False)
		x = x + sc 

		sc = x 
		x = self.ln2(x)
		x = self.l1(x)
		x = self.l2(x)
		x = F.dropout(x, self.drop, self.training, False)
		x = x + sc 
		return x 

class PositionalEmbedding(M.Model):
	def build(self, *inputs):
		indim = inputs[0].shape[2]
		seq_len = inputs[0].shape[1]
		self.embed = Parameter(torch.zeros(1, seq_len, indim))

	def forward(self, x):
		x = self.embed + x 
		return x 

class TransformerNet(M.Model):
	def initialize(self, num_enc, num_heads, dim_per_head, latent_token=True, drop=0.2):
		self.latent_token = latent_token
		self.posemb = PositionalEmbedding()
		self.trans_blocks = nn.ModuleList()
		for i in range(num_enc):
			self.trans_blocks.append(Transformer(num_heads, dim_per_head, drop=drop))

	def build(self, *inputs):
		indim = inputs[0].shape[2]
		if self.latent_token:
			self.token = Parameter(torch.zeros(1, 1, indim))

	def forward(self, x):
		if self.latent_token:
			token = self.token.repeat(x.shape[0], 1, 1)
			x = torch.cat([token, x], dim=1)
		x = self.posemb(x)
		for trans in self.trans_blocks:
			x = trans(x)
		return x 

class ResBlock_v1(M.Model):
	def initialize(self, outchn, stride):
		self.stride = stride
		self.outchn = outchn
		self.bn0 = M.BatchNorm()
		self.c1 = M.ConvLayer(3, outchn, activation=M.PARAM_PRELU, usebias=False, batch_norm=True)
		self.c2 = M.ConvLayer(3, outchn, stride=stride, usebias=False, batch_norm=True)

		# se module 
		#self.c3 = M.ConvLayer(1, outchn//16, activation=M.PARAM_PRELU)
		#self.c4 = M.ConvLayer(1, outchn, activation=M.PARAM_SIGMOID)

		# shortcut 
		self.sc = M.ConvLayer(1, outchn, stride=stride, usebias=False, batch_norm=True)

	def build(self, *inputs):
		self.inchn = inputs[0].shape[1]

	def forward(self, x):
		res = self.bn0(x)
		res = self.c1(res)
		res = self.c2(res)
		# print(res.shape)
		# se
		#se = M.GlobalAvgPool(res)
		#se = self.c3(se)
		#se = self.c4(se)
		#res = res * se 
		# shortcut 
		if self.inchn==self.outchn and self.stride==1:
			sc = x 
		else:
			sc = self.sc(x)
		res = res + sc 
		return res 

class Stage(M.Model):
	def initialize(self, outchn, blocknum, stride):
		self.units = nn.ModuleList()
		for i in range(blocknum):
			self.units.append(ResBlock_v1(outchn, stride=stride if i==0 else 1))
	def forward(self, x):
		for i in self.units:
			x = i(x)
		return x 

class ResNet(M.Model):
	def initialize(self, channel_list, blocknum_list, embedding_size, embedding_bn=True):
		self.c1 = M.ConvLayer(3, channel_list[0], 1, usebias=False, activation=M.PARAM_PRELU, batch_norm=True)
		# self.u1 = ResBlock_v1(channel_list[1], stride=2)
		self.stage1 = Stage(channel_list[1], blocknum_list[0], stride=2)
		self.stage2 = Stage(channel_list[2], blocknum_list[1], stride=2)
		self.stage3 = Stage(channel_list[3], blocknum_list[2], stride=2)
		self.stage4 = Stage(channel_list[4], blocknum_list[3], stride=1)
		self.bn1 = M.BatchNorm()
		print('Embedding_size:', embedding_size)
		self.fc1 = M.Dense(embedding_size, usebias=False)

	def forward(self, x):
		x = self.c1(x)
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		x = self.bn1(x)
		x = M.flatten(x)
		x = F.dropout(x, 0.4, self.training, False)
		x = self.fc1(x)
		return x 

def Res10(dim=512):
	return ResNet([64,64,128,256,512], [2,2,2,2], dim)

class FaceTransNet(M.Model):
	def initialize(self, emb_dim=512):
		self.backbone = Res10()
		self.trans = TransformerNet(num_enc=4, num_heads=8, dim_per_head=64, latent_token=True)
		# whether need last embedding layer 
		self.emb = M.Dense(emb_dim)
	
	def forward(self, x):
		bsize = x.shape[0]
		patches = x.shape[1]
		h = x.shape[3]
		w = x.shape[4]
		# reshape for CNN
		x = x.reshape(bsize*patches, 3, h, w)
		x = self.backbone(x) # b*p, dim
		x = x.reshape(bsize, patches, -1)
		## reshape back for transformer 
		x = self.trans(x)    # [N, seq_len+1, dim]
		x = x[:,0]           # the token corresponds to the classification result
		x = self.emb(x)
		return x 

if __name__=='__main__':
	x = torch.rand(2, 36, 3, 32, 32)
	net = FaceTransNet()
	y = net(x)
	print(y)
	print(y.shape)
