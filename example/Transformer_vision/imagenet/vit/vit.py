import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from TorchSUL import Model as M 

class PatchEmbed(M.Model):
	def initialize(self, patch_size, stride, emb_dim):
		self.c1 = M.ConvLayer(patch_size, emb_dim, stride=stride, pad='VALID', usebias=True)
	def forward(self, x):
		x = self.c1(x)  # [B, d, H, W]
		B, d, h, w = x.shape 
		x = x.reshape(B, d, h*w).transpose(1, 2)
		return x 

class Attention(M.Model):
	def initialize(self, dim, num_heads, attn_drop):
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5 
		self.attn_drop = attn_drop

		self.qkv = M.Dense(dim*3, usebias=True)
		self.proj = M.Dense(dim)

	def forward(self, x, return_attn=False):
		# print(x.shape)
		qkv = self.qkv(x)
		B, N, _ = qkv.shape 
		qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]   # B, n_heads, N, head_dim

		attn = (q @ k.transpose(-1, -2)) * self.scale    # B, n_heads, N, N
		attn = attn.softmax(dim=-1)
		attn = F.dropout(attn, self.attn_drop, self.training, False)

		x = (attn @ v).transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
		x = self.proj(x)
		if return_attn:
			return x, attn
		return x 

class MLP(M.Model):
	def initialize(self, dim, mlp_ratio):
		self.fc1 = M.Dense(dim*mlp_ratio, usebias=True, activation=M.PARAM_GELU)
		self.fc2 = M.Dense(dim)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		return x 

class Block(M.Model):
	def initialize(self, dim, num_heads, mlp_ratio=4, attn_drop=0.0, drop_path=0.0):
		self.drop_path = drop_path
		self.norm1 = M.LayerNorm(1)
		self.attn = Attention(dim, num_heads, attn_drop)
		self.norm2 = M.LayerNorm(1)
		self.mlp = MLP(dim, mlp_ratio)

	def forward(self, x):
		x1 = self.norm1(x)
		x1 = self.attn(x1)
		x1 = F.dropout(x1, self.drop_path, self.training, False)
		x = x1 + x 

		x1 = self.norm2(x)
		x1 = self.mlp(x1)
		x1 = F.dropout(x1, self.drop_path, self.training, False)
		x = x1 + x 
		return x 

	def forward_attn(self, x):
		x1 = self.norm1(x)
		x1, att = self.attn(x1, return_attn=True)
		x1 = F.dropout(x1, self.drop_path, self.training, False)
		x = x1 + x 

		x1 = self.norm2(x)
		x1 = self.mlp(x1)
		x1 = F.dropout(x1, self.drop_path, self.training, False)
		x = x1 + x 
		return x, att

class PosEmbed(M.Model):
	def build_forward(self, x):
		B, N, D = x.shape 
		self.pos_emb = nn.Parameter(torch.zeros(1, N, D))
		nn.init.uniform_(self.pos_emb, -0.2, 0.2)
		return self.pos_emb + x 

	def forward(self, x):
		return self.pos_emb + x 

class Transformer(M.Model):
	def initialize(self, patch_size, patch_stride, emb_dim, depth, num_heads, mlp_ratio=4, attn_drop=0.0, drop_path=0.3):
		self.pe = PatchEmbed(patch_size, patch_stride, emb_dim)
		self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
		nn.init.uniform_(self.cls_token, -0.2, 0.2)
		self.pos_emb = PosEmbed()

		self.blocks = nn.ModuleList()
		for _ in range(depth):
			self.blocks.append(Block(emb_dim, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop_path=drop_path))

	def forward(self, x):
		B = x.shape[0]
		x = self.pe(x)
		cls_token = self.cls_token.expand(B, -1, -1)
		x = torch.cat([cls_token, x], dim=1)
		
		x = self.pos_emb(x)

		for b in self.blocks:
			x = b(x)
		return x[:,0]

	def forward_attn(self, x):
		B = x.shape[0]
		x = self.pe(x)
		cls_token = cls_token.expand(B, 1, 1)
		x = torch.cat([cls_token, x], dim=1)
		x = self.pos_emb(x)

		atts = []
		for b in self.blocks:
			x, att = b.forward_attn(x)
			atts.append(att)
		return x[:,0], atts

class TransNet(M.Model):
	def initialize(self):
		self.trans = Transformer(patch_size=8, patch_stride=8, emb_dim=512, depth=12, num_heads=8)
		self.fc1 = M.Dense(2048, activation=M.PARAM_GELU)
		self.fc2 = M.Dense(512)

	def forward(self, x):
		x = self.trans(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return x 

if __name__=='__main__':
	net = Transformer(patch_size=16, patch_stride=8, emb_dim=512, depth=12, num_heads=8)

	x = torch.zeros(1, 3, 112, 112)
	y = net(x)
	print(y.shape)
