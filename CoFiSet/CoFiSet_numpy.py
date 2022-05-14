import collections
import math
import random
import numpy as np

def pre_procession(train_data_record, test_data_record):
    train_users_max = max(train_data_record[: , 0]) + 1
    train_items_max = max(train_data_record[: , 1]) + 1
    #如果不删除的话，会导致训练集矩阵和测试集矩阵大小不一样
    delete_index = np.where(test_data_record[:, 0] == 4999)[0]
    test_data_record = np.delete(test_data_record, delete_index, 0)
    test_users_max = max(test_data_record[:, 0]) + 1
    test_items_max = max(test_data_record[:, 1]) + 1

    train_users = np.unique(train_data_record[:, 0])
    test_users = np.unique(test_data_record[:, 0])

    I_u = {}
    unI_u = {}
    groundTruth = collections.defaultdict(list)

    train_items = np.unique(train_data_record[:, 1])
    train_data_mat = np.zeros((train_users_max , train_items_max))
    test_data_mat = np.zeros((test_users_max, test_items_max))
    for row in train_data_record:
        train_data_mat[row[0]][row[1]] = 1
    for row in test_data_record:
        test_data_mat[row[0]][row[1]] = 1
    for user in train_users:
        interact_item = train_data_record[:, 1][train_data_record[:, 0] == user]
        unteract_item =np.setdiff1d(train_items, interact_item)
        I_u[user] = interact_item
        unI_u[user] = unteract_item
    for user in test_users:
        test_item = test_data_record[:, 1][test_data_record[:, 0] == user]
        groundTruth[user] = test_item
    return train_data_mat , test_data_mat,train_users_max , train_items_max , test_users_max , test_items_max, I_u, unI_u, train_users, test_users, groundTruth,train_items

def Evaluation_Metircs(predictedIndices,GroundTruth,  topN):
    Precision =0.
    Recall = 0.
    F1 = 0.
    NDCG = 0.
    One_Call = 0.
    sumForPrecision = 0.
    sumForRecall = 0.
    sumForF1 = 0.
    sumForNDCG = 0.
    sumForOne_Call = 0.
    for i in range(len(predictedIndices)):  # for a user,
        if len(GroundTruth[i]) != 0:
            userHit = 0
            dcg = 0
            idcg = 0
            idcgCount = len(GroundTruth[i])
            ndcg = 0
            for j in range(topN):
                if predictedIndices[i][j] in GroundTruth[i]:
                    # if Hit!
                    dcg += 1.0 / math.log2(j + 2)
                    userHit += 1
                if idcgCount > 0:
                    idcg += 1.0 / math.log2(j + 2)
                    idcgCount = idcgCount - 1

            if (idcg != 0):
                ndcg += (dcg / idcg)

            precision_u = userHit / topN
            recall_u = userHit / len(GroundTruth[i])

            sumForPrecision += precision_u
            sumForRecall += recall_u
            if (precision_u + recall_u) != 0:
                sumForF1 += 2 * (precision_u * recall_u / (precision_u + recall_u))
            sumForNDCG += ndcg
            sumForOne_Call += 1 if userHit > 0 else 0
        Precision=sumForPrecision
        Recall=sumForRecall
        F1=sumForF1
        NDCG=sumForNDCG
        One_Call=sumForOne_Call

    return Precision, Recall, F1, NDCG, One_Call


class CoFiSet:
    def __init__(self, train_data_record, train_data_mat, test_data_mat, bi, d, n, m, lr, T, I_u, unI_u):
        self.I_u = I_u
        self.unI_u = unI_u
        self.train_data_mat = train_data_mat
        self.test_data_mat = test_data_mat
        self.bi = bi
        self.a_u = 0.01
        self.a_v = 0.01
        self.beta_v = 0.01
        #init
        self.U = (np.random.random((n , d)) - 0.5) * 0.01
        self.V = (np.random.random((m , d)) - 0.5) * 0.01
        self.d = d
        self.lr = lr
        self.T = T
        self.n = n
        self.m = m
        self.train_data_record = train_data_record
    def sigmoid(self , x):
        return 1 / (1 + np.exp(-x))
    def predict(self, uid, iid):
        r_ui_hat = np.dot(self.U[uid], self.V[iid]) + self.bi[iid]
        return r_ui_hat
    def grandient(self, uid, P, A):
        r_uP = 0
        for i in P:
            r_ui = self.predict(uid, i)
            r_uP += r_ui
        r_uP /= len(P)

        r_uj = np.zeros(len(A))
        for index, j in enumerate(A):
            r_uj[index] = self.predict(uid, j)

        V_p = self.V[P].mean(0)
        #U_u
        temp_u= np.zeros((len(A), self.d))
        for index, j in enumerate(A):
            r_uPj = r_uP - r_uj[index]
            temp_u[index] = -(self.sigmoid(-r_uPj)) * (V_p - self.V[j]) / len(A)
        tri_angle_U_u = temp_u.sum(0) + self.a_u * self.U[uid]
        self.U[uid] -= self.lr * tri_angle_U_u
        #V_i
        temp_v_i = np.zeros((len(A), self.d))
        for index, j in enumerate(A):
            r_uPj = r_uP - r_uj[index]
            temp_v_i[index] = -(self.sigmoid(-r_uPj)) * self.U[uid] / len(P) / len(A)
        tri_angel_V_i_part = temp_v_i.sum(0)
        for i in P:
            tri_angel_V_i = tri_angel_V_i_part + self.a_v * self.V[i]
            self.V[i] -= self.lr * tri_angel_V_i
        #V_j
        for index, j in enumerate(A):
            r_uPj = r_uP - r_uj[index]
            tri_angel_V_j = -(self.sigmoid(-r_uPj)) / len(A) * (-self.U[uid]) + self.a_v * self.V[j]
            self.V[j] -= self.lr * tri_angel_V_j
        #b_i
        temp_b_i = np.zeros(len(A))
        for index, j in enumerate(A):
            r_uPj = r_uP - r_uj[index]
            temp_b_i[index] = -(self.sigmoid(-r_uPj)) / len(A) / len(P)
        tri_angel_b_i_part = temp_b_i.sum()
        for i in P:
            tri_angle_bi = tri_angel_b_i_part + self.beta_v * self.bi[i]
            self.bi[i] -= self.lr * tri_angle_bi
        #b_j
        for index, j in enumerate(A):
            r_uPj = r_uP - r_uj[index]
            tri_angel_b_j = -(self.sigmoid(-r_uPj)) / len(A) * (-1) + self.beta_v * self.bi[j]
            self.bi[j] -= self.lr * tri_angel_b_j

    def train(self):
        for t1 in range(self.T):
            if t1%500 ==0:
                print(t1 , self.T)
            for t2 in range(self.n):
                uid = np.random.choice(train_users, 1)[0]
                select_num = A_P_num if len(self.I_u[uid].tolist()) >= A_P_num else len(self.I_u[uid].tolist())
                P = random.sample(self.I_u[uid].tolist(), select_num)
                A = random.sample(self.unI_u[uid].tolist(), select_num)
                self.grandient(uid , P , A)
            if t1 % 500 == 0:
                r_ui_mat = self.get_r_ui_mat()
                r_ui_mat = np.argsort(r_ui_mat)
                pred = r_ui_mat[:, -5:]
                pre,recal,fone,ndcg,onecall = Evaluation_Metircs(pred, groundTruth, 5)
                print('{:12}{:^12.4f}'.format('Pre@5', pre / len(test_users)))
                print('{:12}{:^12.4f}'.format('Rec@5', recal / len(test_users)))
                print('{:12}{:^12.4f}'.format('F-1@5', fone / len(test_users)))
                print('{:12}{:^12.4f}'.format('NDCG@5', ndcg / len(test_users)))
                print('{:12}{:^12.4f}'.format('1-call@5', onecall / len(test_users)))
    def get_r_ui_mat(self):
        r_ui_mat = np.zeros((self.n , self.m))
        for i in range(self.n):
            for j in range(self.m):
                if self.train_data_mat[i][j]:
                    continue
                r_ui_mat[i][j] = self.predict(i , j)
        return r_ui_mat

if __name__ == '__main__' :
    #Para
    d = 20
    lr = 0.01
    T = 100000
    A_P_num = 3
    #Load Data
    train_data_record = np.loadtxt('XING/copy1.train', dtype=np.int32)
    test_data_record = np.loadtxt('XING/copy1.test', dtype=np.int32)
    train_data_record[: , :2] -= 1
    test_data_record[:, :2] -= 1
    #Data Process
    train_data_mat , test_data_mat, \
    train_users_max , train_items_max , \
    test_users_max , test_items_max, \
    I_u, unI_u , \
    train_users, test_users,\
    groundTruth,train_items = pre_procession(train_data_record, test_data_record)
    #init variable
    b_i = np.zeros(train_items_max)
    for item in range(train_items_max):
        b_i[item] = np.count_nonzero(train_data_mat[:, item]) / len(train_users) - train_data_record.shape[0] / len(train_users) / len(train_items)
    #init model
    cofiset = CoFiSet(train_data_record, train_data_mat, test_data_mat, b_i, d, train_users_max, train_items_max, lr, T, I_u, unI_u)
    #train
    cofiset.train()
    r_ui = cofiset.get_r_ui_mat()

    r_ui_mat = np.argsort(r_ui)
    pred = r_ui_mat[:, -5:]
    pre, recal, fone, ndcg, onecall = Evaluation_Metircs(pred, groundTruth, 5)
    print('{:12}{:^12.4f}'.format('Pre@5', pre / len(test_users)))
    print('{:12}{:^12.4f}'.format('Rec@5', recal / len(test_users)))
    print('{:12}{:^12.4f}'.format('F-1@5', fone / len(test_users)))
    print('{:12}{:^12.4f}'.format('NDCG@5', ndcg / len(test_users)))
    print('{:12}{:^12.4f}'.format('1-call@5', onecall / len(test_users)))
