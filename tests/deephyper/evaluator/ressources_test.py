import collections
import copy

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class Entity:
    def __init__(self, id=None):
        self.id = id
        # ressources should be None or list
        # if its a list, each element should be either a tuple (Entity, int) or an Entity
        # which would be equivalent to (Entity, 1)
        self.ressources = None

    def __repr__(self) -> str:
        repr = f"{type(self).__name__}("
        if self.id is not None:
            repr += f"id={self.id}"
        if self.ressources is not None:
            if self.id is not None:
                repr += ", "
            repr += str(self.ressources)
        repr += ")"
        return repr

    def to_dict(self):
        desc = {}
        if self.id is not None:
            desc["id"] = self.id

        def to_dict(r):
            if type(r) is tuple:
                r, n = r
            else:
                n = 1
            rdict = r.to_dict()
            if type(rdict) is dict:
                k = list(rdict.keys())[0]
                rdict[k]["quantity"] = n
            else:
                rdict = {rdict: {"quantity": n}}
            return rdict

        if self.ressources is not None:
            desc["ressources"] = [to_dict(r) for r in self.ressources]

        if len(desc) > 0:
            fdict = {f"{type(self).__name__}": desc}
        else:
            fdict = f"{type(self).__name__}"

        return fdict

    def to_yaml(self):
        return yaml.dump(self.to_dict(), Dumper=Dumper)

    def number_of(self, entity_class):
        # expoloration of a tree like structure with a queue
        q = collections.deque(self.ressources)
        n = 0
        while len(q) > 0:

            r = q.popleft()
            if type(r) is tuple:
                r, k = r
            else:
                k = 1

            if type(r) is entity_class:
                n += k

            if type(r.ressources) is list and k >= 1:
                q.extend(r.ressources)
        return n

    def __sub__(self, y):
        r, n = self._remove(y)
        return r

    def _remove(self, y):

        if type(y) is tuple:
            y_r, y_n = y
        else:
            y_r, y_n = y, 1

        root_entity = copy.deepcopy(self)

        if root_entity.ressources is None:
            return root_entity

        ressources = []
        for child_entity in root_entity.ressources:

            if type(child_entity) is tuple:
                child_r, child_n = child_entity
            else:
                child_r, child_n = child_entity, 1

            if child_n > 0:
                # if the child_entity and the entity types to remove are same then
                # withdraw them
                if y_r is type(child_r):
                    max_amount = min(child_n, y_n)
                    child_n -= max_amount
                    y_n -= max_amount
                else:  # propagate to child
                    if y_n > 0:
                        child_r, y_n = child_r._remove((y_r, y_n))

                if child_n > 1 or child_n == 0:
                    ressources.append((child_r, child_n))
                elif child_n == 1:
                    ressources.append(child_r)

        root_entity.ressources = ressources

        return root_entity, y_n


class Manager:
    def __init__(self, entity):
        self.entity = copy.deepcopy(entity)

    def request(self, entity_class, quantity=1):
        self.entity = self.entity - ...

    def release(self, entity):
        ...


# Example 1.
def example_1():
    class CPU(Entity):
        def __init__(self):
            super().__init__()

    class Node(Entity):
        def __init__(self, id):
            super().__init__(id)
            self.ressources = [(CPU(), 128)]

    class Cluster(Entity):
        def __init__(self):
            super().__init__("Theta")
            self.ressources = [Node(i) for i in range(8)]

    Theta = Cluster()

    print(Theta)
    print("#Node: ", Theta.number_of(Node))
    print("#CPU: ", Theta.number_of(CPU))

    from pprint import pprint

    theta_as_dict = Theta.to_dict()
    pprint(theta_as_dict)

    output = Theta.to_yaml()

    print(output)


def example_2():
    class CPU(Entity):
        def __init__(self):
            super().__init__()

    class Node(Entity):
        def __init__(self, id):
            super().__init__(id)
            self.ressources = [(CPU(), 128)]

    class Cluster(Entity):
        def __init__(self):
            super().__init__("Theta")
            self.ressources = [Node(i) for i in range(2)]

    Theta = Cluster()
    print("Step 1")
    print(Theta)
    print("#Node: ", Theta.number_of(Node))
    print("#CPU: ", Theta.number_of(CPU))
    print()

    ThetaP = Theta - (CPU, 129)
    print("Step 2")
    print(ThetaP)
    print("#Node: ", ThetaP.number_of(Node))
    print("#CPU: ", ThetaP.number_of(CPU))
    print()

    ThetaP = Theta - Node
    print("Step 3")
    print(ThetaP)
    print("#Node: ", ThetaP.number_of(Node))
    print("#CPU: ", ThetaP.number_of(CPU))

def example_3():

    class Core(Entity):
        def __init__(self):
            super().__init__()

    class GPU(Entity):
        def __init__(self):
            super().__init__()

    class CPU(Entity):

        def __init__(self):
            super().__init__()
            self.ressources = [(Core(), 64)]

    class Node(Entity):
        def __init__(self, id):
            super().__init__(id)
            self.ressources = [(CPU(), 2), (GPU(), 8)]

    class Cluster(Entity):
        def __init__(self):
            super().__init__("ThetaGPU")
            self.ressources = [Node(i) for i in range(2)]

    ThetaGPU = Cluster()
    print(ThetaGPU)
    print("#Node: ", ThetaGPU.number_of(Node))
    print("#CPU: ", ThetaGPU.number_of(CPU))
    print("#Core: ", ThetaGPU.number_of(Core))
    print("#GPU: ", ThetaGPU.number_of(GPU))
    print()
    print(ThetaGPU.to_yaml())

    # ThetaP = Theta - (CPU, 129)
    # print("Step 2")
    # print(ThetaP)
    # print("#Node: ", ThetaP.number_of(Node))
    # print("#CPU: ", ThetaP.number_of(CPU))
    # print()

    # ThetaP = Theta - Node
    # print("Step 3")
    # print(ThetaP)
    # print("#Node: ", ThetaP.number_of(Node))
    # print("#CPU: ", ThetaP.number_of(CPU))

if __name__ == "__main__":
    example_3()