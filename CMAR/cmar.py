from scipy.io import arff
import pandas as pd


# data = arff.loadarff('supermarket.arff')
# print(data)
# df = pd.DataFrame(data[0])

# print(df.head())

rule = 'rule'
f = open(rule)
rules = []
for i in f:
    if 'Instances' in i:
        N = int(i.split(':')[1])
    if 'Attribute' in i:
        A = int(i.split(':')[1])
    if len(i)>1:
        if i[1].isdigit():
            i = i[4:]
            conv = float(i[i.index('conv:(')+ 6:i.index(')', i.index('conv:('))])
            lev = float(i[i.index('lev:(') + 5:i.index(')', i.index('lev:('))])
            lift = float(i[i.index('lift:(') + 6:i.index(')', i.index('lift:('))])
            conf = float(i[i.index('conf:(') + 6:i.index(')', i.index('conf:('))])
            s = i.split(':')
            p = s[0]
            l = s[1].split('==> ')
            c = l[1]
            p_num = int(l[0])
            pc_num = int(s[2].split('<')[0])
            dictionary = {}
            p = p[1:-1]
            p = p.split(',')
            for aa, a in enumerate(p):
                p[aa] = a.split('=')[0]
            c = c[1:-1]
            c = c.split('=')[0].split(" and ")
            dictionary['premise'] = set(p)
            dictionary['conclusion'] = set(c)
            dictionary['premise counter'] = p_num
            dictionary['conclusion counter'] = pc_num
            dictionary['confidence'] = conf
            dictionary['lift'] = lift
            dictionary['lev'] = lev
            dictionary['conv'] = conv
            dictionary['support'] = float("{:.2f}".format(pc_num/N))
            rules.append(dictionary)
print(rules)
redundant = []
for i, element in enumerate(rules):
    for j, elem in enumerate(rules):
        if i<j:
            if element['premise'].issubset(elem['premise']):
                redundant.append(j)


