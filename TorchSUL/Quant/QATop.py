import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


##### START: Quant aware training function
class QATFunc(Function):
    # quant-aware training function 
    @staticmethod 
    def forward(ctx, x, scale, zero_point, Qn, Qp, zero_offset=False, mode='layer_wise', dim=-1):
        ctx.Qn = Qn 
        ctx.Qp = Qp 
        ctx.mode = mode 
        ctx.dim = dim 
        ctx.zero_offset = zero_offset

        if dim!=-1 and mode=='channel_wise':
            # make everything run at last dim, no need manually reshape 
            x = x.transpose(-1, dim)

        zero_point = zero_point.round()
        x_back = x 
        x = x / scale + zero_point
        x = x.round().clamp(Qn, Qp)
        ctx.save_for_backward(x_back, x, scale, zero_point)     # save here to save computation for backward

        x = (x - zero_point) * scale 
        if dim!=-1 and mode=='channel_wise':
            x = x.transpose(-1, dim)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, out_grad):
        out_grad = out_grad.contiguous()
        x, x1, scale, zero_point = ctx.saved_tensors

        # compute ds 
        ds = x1.clone()
        ds -= zero_point
        idx = (x1>ctx.Qn) & (x1<ctx.Qp)
        if ctx.mode=='channel_wise':
            coef = torch.zeros_like(x)
            coef[idx] += x[idx]
            ds -= coef / scale
            ds *= out_grad.transpose(ctx.dim, -1)
        else:
            ds[idx] -= x[idx]/scale
            ds *= out_grad

        # compute db 
        db = torch.zeros_like(x1)
        if not ctx.zero_offset:
            idx = (x1<=ctx.Qn) | (x1>=ctx.Qp)
            if ctx.mode=='channel_wise':
                db[idx] += 1
                db *= -scale
                db *= out_grad.transpose(ctx.dim, -1)
            else:
                db[idx] -= scale
                db *= out_grad

        # compute dx 
        dx = out_grad.clone()
        if ctx.mode=='channel_wise':
            x1 = x1.transpose(ctx.dim, -1)
        dx[(x1<=ctx.Qn) | (x1>=ctx.Qp)] = 0

        if ctx.mode=='channel_wise':
            # use mean here 
            ds = ds.flatten(0,-2).sum(dim=0)
            db = db.flatten(0,-2).sum(dim=0)
        else:
            ds = ds.sum()
            db = db.sum()

        return dx.contiguous(), ds.contiguous(), db.contiguous(), None, None, None, None, None

    @staticmethod
    def symbolic(g, x, scale, zero_point, Qn, Qp, zero_offset=False, mode='layer_wise', dim=-1):
        o1 = g.op('QuantizeLinear', x, scale, zero_point, axis_i=dim)
        return g.op('DequantizeLinear', o1, scale, zero_point, axis_i=dim)
##### END: Quant aware training function

