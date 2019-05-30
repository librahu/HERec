#!/usr/bin/python
#encoding=utf-8
import numpy as np
import time
import random
from math import sqrt,fabs,log
import sys

class HNERec:
    def __init__(self, unum, inum, ratedim, userdim, itemdim, user_metapaths,item_metapaths, trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v):
        self.unum = unum
        self.inum = inum
        self.ratedim = ratedim
        self.userdim = userdim
        self.itemdim = itemdim
        self.steps = steps
        self.delta = delta
        self.beta_e = beta_e
        self.beta_h = beta_h
        self.beta_p = beta_p
        self.beta_w = beta_w
        self.beta_b = beta_b
        self.reg_u = reg_u
        self.reg_v = reg_v

        self.user_metapathnum = len(user_metapaths)
        self.item_metapathnum = len(item_metapaths)

        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, unum)
        print 'Load user embeddings finished.'

        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, inum)
        print 'Load user embeddings finished.'

        self.R, self.T, self.ba = self.load_rating(trainfile, testfile)
        print 'Load rating finished.'
        print 'train size : ', len(self.R)
        print 'test size : ', len(self.T) 

        self.initialize();
        self.recommend();

    def load_embedding(self, metapaths, num):
        X = {}
        for i in range(num):
            X[i] = {}
        metapathdims = []
    
        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/embeddings/' + metapath
            #print sourcefile
            with open(sourcefile) as infile:
                
                k = int(infile.readline().strip().split(' ')[1])
                metapathdims.append(k)
                for i in range(num):
                    X[i][ctn] = np.zeros(k)

                n = 0
                for line in infile.readlines():
                    n += 1
                    arr = line.strip().split(' ')
                    i = int(arr[0]) - 1
                    for j in range(k):
                        X[i][ctn][j] = float(arr[j + 1])
                print 'metapath ', metapath, 'numbers ', n
            ctn += 1
        return X, metapathdims

    def load_rating(self, trainfile, testfile):
        R_train = []
        R_test = []
        ba = 0.0
        n = 0
        user_test_dict = dict()
        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_train.append([int(user)-1, int(item)-1, int(rating)])
                ba += int(rating)
                n += 1
        ba = ba / n
        ba = 0
        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_test.append([int(user)-1, int(item)-1, int(rating)])

        return R_train, R_test, ba

    def initialize(self):
        self.E = np.random.randn(self.unum, self.itemdim) * 0.1
        self.H = np.random.randn(self.inum, self.userdim) * 0.1
        self.U = np.random.randn(self.unum, self.ratedim) * 0.1
        self.V = np.random.randn(self.unum, self.ratedim) * 0.1

        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum
        self.pv = np.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum


        self.Wu = {}
        self.bu = {}
        for k in range(self.user_metapathnum):
            self.Wu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

        self.Wv = {}
        self.bv = {}
        for k in range(self.item_metapathnum):
            self.Wv[k] = np.random.randn(self.itemdim, self.item_metapathdims[k]) * 0.1
            self.bv[k] = np.random.randn(self.itemdim) * 0.1

    def cal_u(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            ui += self.pu[i][k] * (self.Wu[k].dot(self.X[i][k]) + self.bu[k])
        return ui

    def cal_v(self, j):
        vj = np.zeros(self.itemdim)
        for k in range(self.item_metapathnum):
            vj += self.pv[j][k] * (self.Wv[k].dot(self.Y[j][k]) + self.bv[k])
        return vj

    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
        return self.U[i, :].dot(self.V[j, :]) + self.reg_u * ui.dot(self.H[j, :]) + self.reg_v * self.E[i, :].dot(vj)

    def maermse(self):
        m = 0.0
        mae = 0.0
        rmse = 0.0
        n = 0
        for t in self.T:
            n += 1
            i = t[0]
            j = t[1]
            r = t[2]
            r_p = self.get_rating(i, j)

            if r_p > 5: r_p = 5
            if r_p < 1: r_p = 1
            m = fabs(r_p - r)
            mae += m
            rmse += m * m
        mae = mae * 1.0 / n
        rmse = sqrt(rmse * 1.0 / n)
        return mae, rmse

    def recommend(self):
        mae = []
        rmse = []
        starttime = time.clock()
        perror = 99999
        cerror = 9999
        n = len(self.R)

        for step in range(steps):
            total_error = 0.0
            train_start_time = time.time()
            for t in self.R:
                i = t[0]
                j = t[1]
                rij = t[2]

                rij_t = self.get_rating(i, j)
                eij = rij - rij_t
                total_error += eij * eij
                
                U_g = -eij * self.V[j, :] + self.beta_e * self.U[i, :]
                V_g = -eij * self.U[i, :] + self.beta_h * self.V[j, :]

                self.U[i, :] -= delta * U_g
                self.V[j, :] -= delta * V_g

                ui = self.cal_u(i)
                for k in range(self.user_metapathnum):
                    pu_g = self.reg_u * -eij * self.H[j, :].dot(self.Wu[k].dot(self.X[i][k]) + self.bu[k]) + self.beta_p * self.pu[i][k]
                    Wu_g = self.reg_u * -eij * self.pu[i][k] * np.array([self.H[j, :]]).T.dot(np.array([self.X[i][k]])) + self.beta_w * self.Wu[k]
                    bu_g = self.reg_u * -eij * self.pu[i][k] * self.H[j, :] + self.beta_b * self.bu[k]

                    self.pu[i][k] -= 0.1 * self.delta * pu_g
                    self.Wu[k] -= 0.1 * self.delta * Wu_g
                    self.bu[k] -= 0.1 * self.delta * bu_g

                H_g = self.reg_u * -eij * ui + self.beta_h * self.H[j, :]
                self.H[j, :] -= self.delta * H_g

                vj = self.cal_v(j)
                for k in range(self.item_metapathnum):
                    pv_g = self.reg_v * -eij * self.E[i, :].dot(self.Wv[k].dot(self.Y[j][k]) + self.bv[k]) + self.beta_p * self.pv[j][k]
                    Wv_g = self.reg_v * -eij * self.pv[j][k] * np.array([self.E[i, :]]).T.dot(np.array([self.Y[j][k]])) + self.beta_w * self.Wv[k]
                    bv_g = self.reg_v * -eij * self.pv[j][k] * self.E[i, :] + self.beta_b * self.bv[k]

                    self.pv[j][k] -= 0.1 * self.delta * pv_g
                    self.Wv[k] -= 0.1 * self.delta * Wv_g
                    self.bv[k] -= 0.1 * self.delta * bv_g

                E_g = self.reg_v * -eij * vj + 0.01 * self.E[i, :]
                self.E[i, :] -= self.delta * E_g

            perror = cerror
            cerror = total_error / n

            self.delta = 0.93 * self.delta

            if(abs(perror - cerror) < 0.0001):
                break
            print 'step ', step, 'crror : ', sqrt(cerror)
            train_end_time = time.time()
            print 'train time : ', (train_end_time - train_start_time)
            MAE, RMSE = self.maermse()
            mae.append(MAE)
            rmse.append(RMSE)
            #if step % 5 == 0:
            print 'step, MAE, RMSE ', step, MAE, RMSE
            test_time = time.time()
            print 'time: ', test_time - train_end_time
        print 'MAE: ', min(mae), ' RMSE: ', min(rmse)

if __name__ == "__main__":
    unum = 16239
    inum = 14284
    ratedim = 10#int(sys.argv[1])
    userdim = 30
    itemdim = 10
    train_rate = 0.8
    
    user_metapaths = ['ubu', 'ubcibu', 'ubcabu']
    item_metapaths = ['bub', 'bcib', 'bcab']

    for i in range(len(user_metapaths)):
        user_metapaths[i] += '_' + str(train_rate) + '.embedding'
    for i in range(len(item_metapaths)):
        item_metapaths[i] += '_' + str(train_rate) + '.embedding'

    #user_metapaths = ['ubu_' + str(train_rate) +'.embedding', 'ubcibu_''.embedding', 'ubcabu_0.8.embedding']
    
    #item_metapaths = ['bub_0.8.embedding', 'bcib_0.8.embedding', 'bcab_0.8.embedding']
    trainfile = '../data/ub_' + str(train_rate) +'.train'
    testfile = '../data/ub_' + str(train_rate) + '.test'
    steps = 100
    delta = 0.01
    beta_e = 0.1
    beta_h = 0.1
    beta_p = 2
    beta_w = 0.1
    beta_b = 0.01
    reg_u = 1.0
    reg_v = 1.0
    print 'train_rate: ', train_rate
    print 'ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim
    print 'max_steps: ', steps
    print 'delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h, 'beta_p: ', beta_p, 'beta_w: ', beta_w, 'beta_b', beta_b, 'reg_u', reg_u, 'reg_v', reg_v

    HNERec(unum, inum, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v)
