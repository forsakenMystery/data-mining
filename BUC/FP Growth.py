data = []
with open('Dataset.txt', 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        split = line.split(' ')
        li = []
        for j in range(len(split)):
            try:
                n = int(split[j])
                if n < 10:
                    li.append(n)
            except:
                pass
        if len(li) is not 0:
            data.append(li)

# print(data)
# print(len(data))
# for i in range(10):
#     dictionary[i] = 0
# print(dictionary)


def what_item(data_set):
    ret_dict = {}
    for trans in data_set:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


class Node:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.link = None
        self.parent = parent
        self.children = {}

    def increment(self, count):
        self.count += count

    def show(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.show(ind + 1)

    # def __repr__(self):
    #     print("{name:", fp.name, ":", fp.count, "}")
    #     return ""


def create_tree(data_set, threshold=1):
    header = {}
    for trans in data_set:
        for item in trans:
            header[item] = header.get(item, 0) + data_set[trans]
    # print(header) # {0: 512, 1: 512, 3: 512, 4: 512, 5: 512, 6: 512, 7: 512, 8: 512, 2: 512, 9: 512} in shod natijash age hame bashan pas 10 taye aval ke represention ham behtar bashe
    # {0: 7, 1: 2, 3: 4, 4: 8, 5: 4, 6: 3, 7: 5, 8: 3, 2: 2, 9: 2} in mishe 10 taye aval ke migam threshold 4 bashe
    for k in list(header):
        if header[k] < threshold:
            del(header[k])
    frequency = set(header.keys())
    if len(frequency) == 0:
        return None, None
    for k in header:
        header[k] = [header[k], None]
    tree = Node('root', 1, None)
    for tranSet, count in data_set.items():
        dictionary = {}
        for item in tranSet:
            if item in frequency:
                dictionary[item] = header[item][0]
        if len(dictionary) > 0:
            ordered_items_frequency = [v[0] for v in sorted(dictionary.items(), key=lambda p: p[1], reverse=True)]
            next(ordered_items_frequency, tree, header, count)
    return tree, header


def next(items, tree, header, count):
    if items[0] in tree.children:
        tree.children[items[0]].increment(count)
    else:
        tree.children[items[0]] = Node(items[0], count, tree)
        if header[items[0]][1] is None:
            header[items[0]][1] = tree.children[items[0]]
        else:
            header_table(header[items[0]][1], tree.children[items[0]])
    if len(items) > 1:
        next(items[1::], tree.children[items[0]], header, count)


def header_table(source, target):
    while source.link is not None:
        source = source.nodeLink
    source.nodeLink = target


Thresh = 40
iteam = what_item(data[0:100])
# print(initSet)
fp, header = create_tree(iteam, Thresh)
fp.show()
# print(fp.name)
# for i in fp.children:
#     print(i)


