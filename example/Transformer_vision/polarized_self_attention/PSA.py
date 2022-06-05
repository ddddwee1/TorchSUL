import torch
import torch.nn as nn
import TorchSUL.Model as M

class PSA_CH(M.Model):
	# channel only self attention
	def initialize(self, attn_ratio=0.5):
		# attn_ratio: attention ratio 
		self.attn_ratio = attn_ratio

	def build_forward(self x):
		# forward function for building this class
		attn_ratio = self.attn_ratio
		attn_chn = x.shape[1]
		self.wq = M.ConvLayer(1)
		self.wv = M.ConvLayer(int(attn_chn * attn_ratio))
		self.wout = M.Dense(out_chn)
		self.ln = nn.LayerNorm(out_chn)
		self.forward(x)

	def forward(self, x): 
		inp = x
		N,_,h,w = x.shape
		q = self.wq(x)   # [N,1,h,w]
		v = self.wv(x)   # [N,C*ratio,h,w]
		q = q.reshape(N, h*w)
		q = torch.softmax(q, dim=1)
		v = v.reshape(N, -1, h*w)  
		z = torch.einsum('ij,ikj->ik', q, v)   # [N,C*ratio]
		z = self.wout(z)
		z = self.ln(z)
		z = torch.sigmoid(z)   # [N,C]
		out = inp * z.unsqueeze(-1).unsqueeze(-1)
		return out 

class PSA_SP(M.Model):
	# spatial only self attention
	def initialize(self, attn_ratio):
		self.attn_ratio = attn_ratio

	def build_forward(self, x):
		attn_ratio = self.attn_ratio
		attn_chn = x.shape[1]
		self.wq = M.ConvLayer(int(attn_chn * attn_ratio))
		self.wv = M.ConvLayer(int(attn_chn * attn_ratio))
		self.forward(x)

	def forward(self, x):
		inp = x 
		q = self.wq(x)
		v = self.wv(x)          # [N,C*ratio,h,w]
		q = q.mean(dim=(2,3))   # [N,C*ratio]
		q = torch.softmax(q, dim=1)
		z = torch.einsum('ijkl,ij->ikl', v,q) # [N,h,w]
		z = z.unsqueeze(1)    # [N,1,h,w]
		z = torch.sigmoid(z)
		out = inp * z 
		return out 

