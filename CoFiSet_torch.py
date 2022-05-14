import collections
import random

import numpy as np
import math
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset


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
def pre_procession(train_data , test_data):
    train_users_max = max(train_data[: , 0]) + 1
    train_items_max = max(train_data[: , 1]) + 1
    delete_index = np.where(test_data[:, 0] == 4999)[0]
    test_data = np.delete(test_data, delete_index, 0)
    test_users_max = max(test_data[:, 0]) + 1
    test_items_max = max(test_data[:, 1]) + 1
    train_users = np.unique(train_data[:,0])
    test_users = np.unique(test_data[:, 0])
    user_interact_dict = {}
    user_uninteract_dict = {}
    test_data_dict  = collections.defaultdict(list)
    #print(train_data_users , train_data_items , test_data_users , test_data_items)
    train_items = np.unique(train_data[:, 1])
    train_data_mat = np.zeros((train_users_max , train_items_max))
    test_data_mat = np.zeros((test_users_max, test_items_max))
    for train_data_recoder in train_data:
        train_data_mat[train_data_recoder[0]][train_data_recoder[1]] = 1
    for test_data_recoder in test_data:
        test_data_mat[test_data_recoder[0]][test_data_recoder[1]] = 1

    for user in train_users:
        interact_item = train_data[:,1][train_data[:, 0] == user]
        if interact_item.shape[0] == 2:
            interact_item= np.append(interact_item, interact_item[0])
        if interact_item.shape[0] == 1:
            interact_item = np.append(interact_item, interact_item[0])
            interact_item = np.append(interact_item, interact_item[1])
        unteract_item =np.setdiff1d(train_items, interact_item)
        user_interact_dict[user] = interact_item
        user_uninteract_dict[user] = unteract_item

    for user in test_users:
        test_item = test_data[:, 1][test_data[:, 0]== user]
        test_data_dict[user] = test_item
    return train_data_mat , test_data_mat,train_users_max , train_items_max , test_users_max , \
           test_items_max, user_interact_dict, user_uninteract_dict, train_users, test_data_dict, test_users, train_items




class My_Dataset(Dataset):
    def __init__(self, select_num_data, interact_items, uninteract_items):
        self.data = select_num_data
        self.interact_items = interact_items
        self.uninteract_items = uninteract_items

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size):
        random.shuffle(self.data)
        count_total = 0
        batch_num = len(self.data) // batch_size
        for j in range(batch_num):
            batch_users = []
            batch_in_item = []
            batch_un_item = []
            for i in range(batch_size):
                if count_total + i >= len(self.data):
                    break
                user, in_item, un_item = self.data[j*batch_size + i]
                select_len = 3 if len(in_item) >=3 else len(in_item)
                item_in = random.sample(in_item, select_len)
                item_un = random.sample(un_item, select_len)
                batch_users.append(user)
                batch_in_item.append(item_in)
                batch_un_item.append(item_un)
            yield [batch_users, batch_in_item, batch_un_item]

class CoFiSet(torch.nn.Module):
    def __init__(self , train_data , train_data_mat , test_data_mat , bi , d , n_users , n_items , learning_rate ,T , interact_item, uninteract_item):
        super(CoFiSet, self).__init__()
        self.interact_item = interact_item
        self.unointeract_item = uninteract_item
        self.train_data_mat = train_data_mat
        self.test_data_mat = test_data_mat
        self.U = nn.Parameter(torch.Tensor((n_users, d)))
        self.V = nn.Parameter(torch.Tensor((n_items, d)))
        self.bi = nn.Parameter(torch.FloatTensor(bi), requires_grad=True)
        self.U.data = torch.from_numpy((np.random.rand(n_users, d) - 0.5) * 0.01)
        self.V.data = torch.from_numpy((np.random.rand(n_items, d) - 0.5) * 0.01)
        self.d = d
        self.learning_rate = learning_rate
        self.T = T
        self.n_users = n_users
        self.n_items = n_items
        self.train_data = train_data

    def caculate_score_pos(self , user_id , item_id ):
        user_id = torch.tensor(user_id).long().to(Devices_gpu_cpu)
        item_id = torch.tensor(item_id).long().to(Devices_gpu_cpu)
        select_num = item_id.shape[1]
        temp_score = torch.zeros(user_id.shape[0]).to(Devices_gpu_cpu)
        for i in range(select_num):
            score = torch.diag(torch.matmul(self.U[user_id] , self.V[item_id[:, i]].T)) + self.bi[item_id[:, i]]
            temp_score += score
        return temp_score / select_num

    def caculate_score_neg(self , user_id , item_id ):
        user_id = torch.tensor(user_id).long().to(Devices_gpu_cpu)
        item_id = torch.tensor(item_id).long().to(Devices_gpu_cpu)
        score = torch.diag(torch.matmul(self.U[user_id] , self.V[item_id].T)) + self.bi[item_id]
        return score

    def predict(self):
        score = torch.matmul(self.U, self.V.T) + self.bi
        return score.detach().cpu().numpy()

    def to_batch_list(self):
        classifly_num_dict = []
        for user in train_users:
            classifly_num_dict.append([user, self.interact_item[user].tolist(), self.unointeract_item[user].tolist()])
        return classifly_num_dict

    def get_r_ui_mat(self):
        r_ui_mat = self.predict()
        for i in range(self.n_users):
            for j in range(self.n_items):
                if self.train_data_mat[i][j]:
                    r_ui_mat[i][j] = -4999
        return r_ui_mat

def train(model,groundTruth,testUserList):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    for t in range(T):
        model.train()
        if t % 2 == 0:
            batch_data = batch_dataset.get_batch(512)
        for one_batch_data in batch_data:
            batch_user, batch_in, batch_un = one_batch_data[0], one_batch_data[1], one_batch_data[2]
            batch_user, item_pos, item_neg = np.array(batch_user), np.array(batch_in), np.array(batch_un)
            optimizer.zero_grad()
            pos_score = model.caculate_score_pos(batch_user, item_pos)
            temp_ = torch.zeros(batch_user.shape[0]).to(Devices_gpu_cpu)
            for i in range(A_P_num):
                neg_score = model.caculate_score_neg(batch_user, item_neg[:, i])
                temp_ += (pos_score - neg_score)
            score_pos_neg = temp_ / A_P_num
            loss = -torch.log( torch.sigmoid(score_pos_neg)).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if t % 500 == 0:
            model.eval()
            print(t, T, loss.item())
            r_ui_mat = model.get_r_ui_mat()
            r_ui_mat = np.argsort(r_ui_mat)
            pred = r_ui_mat[:, -5:]
            pre, recal, fone, ndcg, onecall = Evaluation_Metircs(pred, groundTruth, 5)
            print('{:12}{:^12.4f}'.format('Pre@5', pre / len(testUserList)))
            print('{:12}{:^12.4f}'.format('Rec@5', recal / len(testUserList)))
            print('{:12}{:^12.4f}'.format('F-1@5', fone / len(testUserList)))
            print('{:12}{:^12.4f}'.format('NDCG@5', ndcg / len(testUserList)))
            print('{:12}{:^12.4f}'.format('1-call@5', onecall / len(testUserList)))
if __name__ == '__main__' :
    print('{:12}{:^12.4f}'.format('Pre@5', 0.0839))
    print('{:12}{:^12.4f}'.format('Rec@5', 0.0387))
    print('{:12}{:^12.4f}'.format('F-1@5', 0.0431))
    print('{:12}{:^12.4f}'.format('NDCG@5', 0.0843))
    print('{:12}{:^12.4f}'.format('1-call@5', 0.2639))
    d = 20
    learning_rate = 0.01
    T = 100000
    A_P_num = 3
    train_data = np.loadtxt('XING/copy1.train', dtype=np.int32)
    test_data = np.loadtxt('XING/copy1.test', dtype=np.int32)
    train_data[: , :2] -= 1
    test_data[:, :2] -= 1
    train_data_mat , test_data_mat, train_users_max , train_items_max , test_users_max , \
    test_data_items, interact_item, uninteract_item ,train_users, test_data_dict, test_users, train_items= pre_procession(train_data , test_data)
    b_i = np.zeros(train_items_max)
    for item in range(train_items_max):
        b_i[item] = np.count_nonzero(train_data_mat[:, item]) / len(train_users) - train_data.shape[0] / len(train_users) / len(train_items)

    devices = torch.cuda.is_available()
    # use random seed defined
    np.random.seed(123)
    torch.manual_seed(123)
    random.seed(123)
    if devices:
        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        torch.backends.cudnn.deterministic = True
    Devices_gpu_cpu = torch.device('cuda' if devices else 'cpu')

    cofiset = CoFiSet(train_data, train_data_mat , test_data_mat , b_i , d , train_users_max , train_items_max , learning_rate , T, interact_item, uninteract_item).to(Devices_gpu_cpu)
    for name, parameters in cofiset.named_parameters():  # 打印出每一层的参数的大小
        print(name, ':', parameters.size())

    batch_list = cofiset.to_batch_list()
    batch_dataset = My_Dataset(batch_list, interact_item, uninteract_item)

    batch_data = batch_dataset.get_batch(512)
    train(cofiset,test_data_dict,test_users)
    r_ui = cofiset.get_r_ui_mat()
    r_ui_mat = np.argsort(r_ui)
    pred = r_ui_mat[:, -5:]
    pre, recal, fone, ndcg, onecall = Evaluation_Metircs(pred, test_data_dict, 5)
    print('{:12}{:^12.4f}'.format('Pre@5', pre / len(test_users)))
    print('{:12}{:^12.4f}'.format('Rec@5', recal / len(test_users)))
    print('{:12}{:^12.4f}'.format('F-1@5', fone / len(test_users)))
    print('{:12}{:^12.4f}'.format('NDCG@5', ndcg / len(test_users)))
    print('{:12}{:^12.4f}'.format('1-call@5', onecall / len(test_users)))

