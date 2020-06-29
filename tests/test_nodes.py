import torch

from vibrant.nodes import Nodes


class TestNodes:
    def test_default_state(self):
        nodes = Nodes(torch.rand(10, 2))
        assert nodes.X.size() == nodes.u.size()
        assert nodes.X.size() == nodes.v.size()
        assert torch.allclose(nodes.u, torch.zeros(10, 2))
        assert torch.allclose(nodes.v, torch.zeros(10, 2))

    def test_len(self):
        nodes = Nodes(torch.rand(10, 2))
        assert len(nodes) == 10

    def test_current_position(self):
        nodes = Nodes(torch.rand(10, 2))
        assert torch.allclose(nodes.x(), nodes.X)
        nodes.u = torch.rand(10, 2)
        assert torch.allclose(nodes.x(), nodes.X + nodes.u)
