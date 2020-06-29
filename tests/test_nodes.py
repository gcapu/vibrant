import torch

from vibrant.nodes import Nodes


class TestNodes:
    def produces_correct_default_state(self):
        nodes = Nodes(torch.rand(10, 2))
        assert nodes.X.size() == nodes.u.size()
        assert nodes.X.size() == nodes.v.size()

    def len_works(self):
        nodes = Nodes(torch.rand(10, 2))
        assert len(nodes) == 10

    def obtains_current_position(self):
        nodes = Nodes(torch.rand(10, 2))
        assert torch.allclose(nodes.x(), nodes.X)
        nodes.u = torch.rand(10, 2)
        assert torch.allclose(nodes.x(), nodes.X + nodes.u)
