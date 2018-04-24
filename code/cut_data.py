#!/user/bin/python
import random

train_rate = 0.8

R = []
with open('../data/ub.txt', 'r') as infile:
    for line in infile.readlines():
        user, item, rating = line.strip().split('\t')
        R.append([user, item, rating])

random.shuffle(R)
train_num = int(len(R) * train_rate)

with open('../data/ub_' + str(train_rate) + '.train', 'w') as trainfile,\
     open('../data/ub_' + str(train_rate) + '.test', 'w') as testfile:
     for r in R[:train_num]:
         trainfile.write('\t'.join(r) + '\n')
     for r in R[train_num:]:
         testfile.write('\t'.join(r) + '\n')


