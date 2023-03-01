import os
import os.path as osp

from hpcs.utils.data import get_hierarchy_path

HIERARCHY_ROOT = get_hierarchy_path()


def get_hierarchy_list(category, levels):
    leaves, leaf_nodes, lines_hier = get_leaves(category)
    hierarchy_list = []
    for level in levels:
        with open(os.path.join(HIERARCHY_ROOT, '%s-level-%d.txt' % (category, level)), 'r') as fin:
            lines_level = fin.readlines()
            hierarchy_level = get_hierarchy_level(leaves, lines_level, lines_hier)
            hierarchy_list.append(hierarchy_level)
    hierarchy_list_remap = remap_leaves(hierarchy_list)
    return hierarchy_list_remap


def get_leaves(category):
    with open(os.path.join(HIERARCHY_ROOT, '%s.txt' % (category)), 'r') as fin:
        lines_hier = fin.readlines()
        leaves = []
        leaf_nodes = []
        for index, line in enumerate(lines_hier):
            if 'leaf' in line:
                leaves.append(index + 1)
                leaf_nodes.append([index + 1])
    return leaves, leaf_nodes, lines_hier


def get_hierarchy_level(leaf_nodes, lines_level, lines_hier):
    numbers = []
    for line in lines_level:
        number = line[:2]
        numbers.append(int(number))
    numbers.append(len(lines_hier) + 1)
    level_numbers = []
    for index, item in enumerate(numbers):
        if index != len(numbers) - 1:
            level = list(range(item, numbers[index + 1]))
            if level == []:
                level = [item]
            level_numbers.append(level)
    final = []
    for item in level_numbers:
        identical = list(set(item) & set(leaf_nodes))
        final.append(sorted(identical))
    return final


def remap_leaves(original):
    for index1, branch in enumerate(original):
        i = 0
        for index2, channel in enumerate(branch):
            for index3, leaf in enumerate(channel):
                original[index1][index2][index3] = i
                i += 1
    return original
