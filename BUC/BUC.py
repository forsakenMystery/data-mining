import copy
from functools import reduce
from itertools import chain, combinations


def powerset(s):
    x = len(s)
    ret =[]
    for i in range(1, 1 << x):
        ret.append([s[j] for j in range(x) if (i & (1 << j))])
    return ret


def get_count(elements, cond=None):
    if cond:
        cnt = sum(cond(elem) for elem in elements)
    else:
        cnt = len(elements)
    return cnt


column = ["A", "B", "C", "D", "E"]
ps = lambda s: reduce(lambda P, x: P + [subset | {x} for subset in P], s, [set()])
power = powerset(column)

min = 2
fun = [lambda x: 0 <= x < 4, lambda x: 4 <= x < 8, lambda x: 8 <= x < 12, lambda x: 12 <= x < 16, lambda x: 16 <= x < 20]
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
                li.append(n)
            except:
                pass
        data.append(li)

data_set = []
for i in range(len(data)):
    set_to_append = {}
    for j in range(len(column)):
        set_to_append[column[j]] = get_count(data[i], fun[j])
    data_set.append(set_to_append)
# print(data_set)

# A = []
# for i in range(len(data_set)):
#     dat = data_set[i]
#     if dat["A"] > 2 and dat["B"] > 2:
#         A.append(i)
# print(len(A))


def get_next_data_layer_by_min_sup(data_set, input_data, dim, min_sup):
    if len(input_data) == 0 and dim == 1:
        ans = {}
        for i in range(len(data_set)):
            for j in range(len(column)):
                if data_set[i][column[j]] > min_sup:
                    if column[j] in ans:
                        ans[column[j]].append(i)
                    else:
                        ans[column[j]] = [i]
        return [ans]
    else:
        for i in range(len(input_data)):
            pass

    return bs


def buc(data_set, input_list, dim, min_sup=2):
    if dim < len(column):
        dim = dim + 1
        level_output = get_next_data_layer_by_min_sup(data_set, input_list, dim, min_sup)
        return buc(data_set, level_output, dim, min_sup)
    else:
        return input_list


# buc_output = buc(data_set, [], 0, min)
# print(buc_output)

dic = {}
for i in range(len(data_set)):
    # print(data_set[i])
    for se in power:
        flag = True
        for char in se:
            # print(char)
            if data_set[i][char] > min:
                # print("yes")
                flag = True
            else:
                # print("shit")
                flag = False
                break
        # print("====================")
        if flag:
            if tuple(se) in dic:
                dic[tuple(se)].append(i)
            else:
                dic[tuple(se)] = [i]
for se in power:
    print(se, ">", min, "has", len(dic[tuple(se)]))
