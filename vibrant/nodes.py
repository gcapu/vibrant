import torch


class Nodes:
    "Nodes"

    def __init__(self, X, u=None, v=None):
        self.X = X
        self.u = u if u is not None else torch.zeros_like(X)
        self.v = v if v is not None else torch.zeros_like(X)
        self.m = None
        self.f = None

    def __len__(self):
        return len(self.X)

    def x(self):
        return self.X + self.u
