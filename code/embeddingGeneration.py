import os

train_rate = 0.8
dim = 128
walk_len = 5
win_size = 3
num_walk = 10

metapaths = ['ubu', 'ubcabu', 'ubcibu', 'bub', 'bcab', 'bcib']

for metapath in metapaths:
	metapath = metapath + '_' + str(train_rate)
	input_file = '../data/metapath/' + metapath + '.txt'
	output_file = '../data/embeddings/' + metapath + '.embedding'

	cmd = 'deepwalk --format edgelist --input ' + input_file + ' --output ' + output_file + \
	      ' --walk-length ' + str(walk_len) + ' --window-size ' + str(win_size) + ' --number-walks '\
	       + str(num_walk) + ' --representation-size ' + str(dim)

	print cmd;
	os.system(cmd)
