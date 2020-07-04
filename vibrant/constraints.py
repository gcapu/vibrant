import torch


class ImposeVelocity:
    """Constrain the velocity of nodes to a specified value."""

    def __init__(self, nodes, node_ids, velocity):
        self.nodes = nodes
        self.node_ids = node_ids
        self.velocity = torch.as_tensor(velocity)
        if not torch.is_floating_point(self.velocity):
            self.velocity = torch.as_tensor(velocity, dtype=torch.float)

    def __call__(self, field="v"):
        if field == "v":
            self.nodes.v[self.node_ids] = self.velocity
