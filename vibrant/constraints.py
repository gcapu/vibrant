import torch


class ImposeVelocity:
    """Constrain the velocity of nodes to a specified value."""

    def __init__(self, node_ids, velocity):
        self.node_ids = node_ids
        self.velocity = torch.as_tensor(velocity)
        if not torch.is_floating_point(self.velocity):
            self.velocity = torch.as_tensor(velocity, dtype=torch.float)

    def __call__(self, nodes, field="v"):
        if field == "v":
            nodes.v[self.node_ids] = self.velocity
