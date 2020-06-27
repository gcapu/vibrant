import torch
from vibrant.nodes import Nodes


class Model:
    """ Finite element model

    Container for the nodes elements, loads and constraints. It also stores some model 
        properties like the time and damping factor. 

    Args:
        nodes (Nodes or 2D tensor): The nodes of the model. If it is a tensor, it 
            corresponds to the initial position of the nodes.
        elements (Elements): 

    """

    def __init__(self, nodes=None, elements=None, time=0, damping=0):

        self.nodes = Nodes(nodes) if isinstance(nodes, torch.Tensor) else nodes
        self.elements = [] if elements is None else [elements]
        self.loads = []
        self.constraints = []
        self.damping = damping
        self.time = time

    def mass(self):
        """Update and return the mass"""
        if self.nodes.m is None:
            self.nodes.m = sum(els.mass() for els in self.elements)
        return self.nodes.m

    def force(self):
        # Update nodal forces
        internal_force = sum(els.force() for els in self.elements)
        external_force = sum(load.force() for load in self.loads)
        self.nodes.f = internal_force + external_force
        if self.damping:
            self.nodes.f = -self.damping * self.mass() * self.nodes.v
        return self.nodes.f

    def acceleration(self):
        """Update the mass and the force, and calculate the acceleration"""
        # return nodal acceleration
        return self.force() / self.mass()

