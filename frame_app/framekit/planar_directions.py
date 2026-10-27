import copy


def clone_geom_for_direction(geom, direction: str):
    g = copy.deepcopy(geom)
    g.direction = direction.upper()
    return g


def clone_nodes_for_direction(nodes, direction: str):
    # geometry is identical for planar frames
    return copy.deepcopy(nodes)


def clone_elements_for_direction(elements, direction: str):
    return copy.deepcopy(elements)


def label_direction(direction: str):
    return direction.upper()