#!/usr/bin/python
import sys
import numpy as np
import random

class metapathGeneration:
    def __init__(self, unum, bnum, conum, canum, cinum):
        self.unum = unum + 1
        self.bnum = bnum + 1
        self.conum = conum + 1
        self.canum = canum + 1
        self.cinum = cinum + 1
        ub = self.load_ub('../data/ub_0.8.train')
        self.get_UBU(ub, '../data/metapath/ubu_0.8.txt')
        self.get_UBCaBU(ub, '../data/bca.txt', '../data/metapath/ubcabu_0.8.txt')
        self.get_UBCiBU(ub, '../data/bci.txt', '../data/metapath/ubcibu_0.8.txt')
        self.get_BUB(ub, '../data/metapath/bub_0.8.txt')
        self.get_BCiB('../data/bci.txt', '../data/metapath/bcib_0.8.txt')
        self.get_BCaB('../data/bca.txt', '../data/metapath/bcab_0.8.txt')

    def load_ub(self, ubfile):
        ub = np.zeros((self.unum, self.bnum))
        with open(ubfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                ub[int(user)][int(item)] = 1 
        return ub
    
    def get_UCoU(self, ucofile, targetfile):
        print 'UCoU...'
        uco = np.zeros((self.unum, self.conum))
        with open(ucofile, 'r') as infile:
            for line in infile.readlines():
                u, co, _ = line.strip().split('\t')
                uco[int(u)][int(co)] = 1

        uu = uco.dot(uco.T)
        print uu.shape
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print 'total = ', total
    
    def get_UU(self, uufile, targetfile):
        print 'UU...'
        uu = np.zeros((self.unum, self.unum))
        with open(uufile, 'r') as infile:
            for line in infile.readlines():
                u1, u2, _ = line.strip().split('\t')
                uu[int(u1)][int(u2)] = 1
        r_uu = uu.dot(uu.T)

        print r_uu.shape
        print 'writing to file...'
        total = 0 
        with open(targetfile, 'w') as outfile:
            for i in range(r_uu.shape[0]):
                for j in range(r_uu.shape[1]):
                    if r_uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(r_uu[i][j])) + '\n')
                        total += 1
        print 'total = ', total
                                                                                                                                     

    def get_UBU(self, ub, targetfile):
        print 'UMU...'

        uu = ub.dot(ub.T)
        print uu.shape
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print 'total = ', total
    
    def get_BUB(self, ub, targetfile):
        print 'MUM...'
        mm = ub.T.dot(ub)
        print mm.shape
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print 'total = ', total
    
    def get_BCiB(self, bcifile, targetfile):
        print 'BCiB..'

        bci = np.zeros((self.bnum, self.cinum))
        with open(bcifile) as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                bci[int(m)][int(d)] = 1

        mm = bci.dot(bci.T)
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0])[1:]:
                for j in range(mm.shape[1])[1:]:
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print 'total = ', total

    def get_BCaB(self, bcafile, targetfile):
        print 'BCaB..'

        bca = np.zeros((self.bnum, self.canum))
        with open(bcafile) as infile:
            for line in infile.readlines():
                m, a,__ = line.strip().split('\t')
                bca[int(m)][int(a)] = 1

        mm = bca.dot(bca.T)
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0])[1:]:
                for j in range(mm.shape[1])[1:]:
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print 'total = ', total
    
    def get_MTM(self, mtfile, targetfile):
        print 'MTM..'

        mt = np.zeros((self.mnum, self.tnum))
        with open(mtfile) as infile:
            for line in infile.readlines():
                m, a,__ = line.strip().split('\t')
                mt[int(m)][int(a)] = 1

        mm = mt.dot(mt.T)
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0])[1:]:
                for j in range(mm.shape[1])[1:]:
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print 'total = ', total
    
    def get_UBCaBU(self, ub, bcafile, targetfile):
        print 'UBCaBU...'

        bca = np.zeros((self.bnum, self.canum))
        with open(bcafile, 'r') as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                bca[int(m)][int(d)] = 1

        uu = ub.dot(bca).dot(bca.T).dot(ub.T)
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print 'total = ', total
    
    def get_UBCiBU(self, ub, bcifile, targetfile):
        print 'UBCiBU...'

        bci = np.zeros((self.bnum, self.cinum))
        with open(bcifile, 'r') as infile:
            for line in infile.readlines():
                m, a, _ = line.strip().split('\t')
                bci[int(m)][int(a)] = 1

        uu = ub.dot(bci).dot(bci.T).dot(ub.T)
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print 'total = ', total

if __name__ == '__main__':
    #see __init__() 
    metapathGeneration(unum=16239, bnum=14284, conum=11, canum=511, cinum=47)
