import torch


class World:
    def __init__(self):
        self.objects = []

    def add(self, o):
        self.objects.append(o)

    def clear(self):
        self.objects = []

    def hit(self, r_in):
        distances = [obj.intersect(r_in) for obj in self.objects]  # hit should return the distance to the object
        nearest = torch.min(torch.stack(distances, dim=0), dim=0).values

        return distances, nearest
