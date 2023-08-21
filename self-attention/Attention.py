class Self_Attention(nn.Module):
    def __init__(self, dim, dk, dv):    # dim=2, dk=2, dv=3
        super(Self_attention, slef).__init()
        self.scale = dk ** -0.5
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)

    def forward(self, x):
        q = self.q(x)   # q = {Tensor: (1, 4, 2)}
        k = self.k(x)   # k = {Tensor: (1, 4, 2)}
        v = self.v(x)   # k = {Tensor: (1, 4, 3)}

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        return x

att = Self_Attention(dim=2, dk=2, dv=3)
x = torch.rand((1, 4, 2))
output = att(x)
