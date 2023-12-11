import torch
from torch import nn
from PromptUNet.PromptEncoder import *
class ResidualCrossConnection(ResidualConnection):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
            super().__init__(d_model,dropout)

    def forward(self,x,y,sublayer):
        output = self.dropout(sublayer(x,y))
        return self.norm(x + output)

class Embeddings(nn.Module):

    def __init__(self,d_model, size,device):
        super().__init__()
        self.d_model = d_model
        self.num_labels = size[0]*size[1]
        self.embedding = nn.Embedding(self.num_labels,d_model,device=device)

    def forward(self,x):
        return self.embedding(x)
class MultiHeadCrossAttentionLayer(nn.Module):
    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()

        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.attn_layer = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)



    def forward(self, q_tensors,kv_tensors):  # x_shape = B,L+1,d_model

        Q = self.w_q(q_tensors)
        K = self.w_k(kv_tensors)
        V = self.w_v(kv_tensors)
        attn_output = self.attn_layer(Q, K, V, need_weights=False)

        return attn_output[0]

class CrossAttentionBlock(nn.Module):

    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()
        self.CrossAttention1 = MultiHeadCrossAttentionLayer(d_model,num_heads,dropout)
        self.FFN = FFN(d_model,dropout)
        self.res_connection1 = ResidualCrossConnection(d_model,dropout)
        self.res_connection2 = ResidualConnection(d_model,dropout)
        self.CrossAttention2 = MultiHeadCrossAttentionLayer(d_model,num_heads,dropout)
        self.FFN2 = FFN(d_model, dropout)
        self.res_connection3 = ResidualCrossConnection(d_model, dropout)
        self.res_connection4 = ResidualConnection(d_model, dropout)

    def forward(self,images,prompts):
        prompts = self.res_connection1(prompts,images,self.CrossAttention1)
        prompts = self.res_connection2(prompts,self.FFN)
        images = self.res_connection3(images,prompts,self.CrossAttention2)
        images = self.res_connection4(images,self.FFN2)

        return images,prompts

class ImageEncoder(nn.Module):

    def __init__(self,device,d_model=384, num_heads=8,num_blocks=6, dropout=0.1,size=(32,32)):
        super().__init__()
        self.device = device
        self.patch_embeddings = Embeddings(d_model,size,self.device)
        self.cross_attns = nn.ModuleList([CrossAttentionBlock(d_model,num_heads,dropout) for _ in range(num_blocks)])
        self.size = size
    def forward(self,images_input,prompts):
        images_input = images_input.view(images_input.shape[0],images_input.shape[1],-1)
        B,D,L = images_input.shape
        indices = torch.arange(L).unsqueeze(0).repeat(B, 1)
        indices = indices.to(self.device)
        images_input = torch.transpose(images_input,1,2)
        images = images_input + self.patch_embeddings(indices) # += is an in place alteration and messes up backprop


        for i in range(len(self.cross_attns)):
            images,prompts = self.cross_attns[i](images,prompts)

        images = torch.reshape(images,(images.shape[0],images.shape[2],self.size[0],self.size[1]))

        return images
