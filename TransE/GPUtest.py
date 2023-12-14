import numpy as np
import cupy as cp
from scipy.spatial.distance import cdist
import random
import pickle

# Hyperparameters
# 学习率
learning_rate = 0.01
# 实体、关系维度
embedding_dim = 50
# 正例、负例边界
margin = 1.0
# 训练轮次
epochs = 100
# 训练批次
batch_size = 128


class TransE:
    def __init__(self, entity2id, relation2id, train_triples, valid_triples, test_triples):
        # Extract unique entities and relations
        self.entities = set(entity2id.keys())
        self.relations = set(relation2id.keys())
        # Create dictionaries to map entities and relations to unique IDs
        self.entity2id, self.relation2id = entity2id, relation2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}

        # Initialize entity and relation embeddings randomly
        self.entity_embeddings = cp.random.randn(len(self.entities), embedding_dim)
        self.relation_embeddings = cp.random.randn(len(self.relations), embedding_dim)

        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        self.batch_size = int(len(train_triples) / 128)
        self.learning_rate = learning_rate
        self.margin = margin

    def predict(self, head, relation, tail):
        # Look up the embeddings for the entities and relation
        e1 = self.entity_embeddings[self.entity2id[head]]
        rel = self.relation_embeddings[self.relation2id[relation]]
        e2 = self.entity_embeddings[self.entity2id[tail]]

        # If the score is above the threshold, predict 1 (true), otherwise predict 0 (false)
        distances = cdist(self.entity_embeddings.get(), cp.reshape(e1 + rel, (1, -1)).get())
        # 如果有相同距离最短的向量，随机选择一个
        min_indexes = cp.argwhere(distances == cp.min(distances)).flatten()
        if len(min_indexes) > 1:
            index = cp.random.choice(min_indexes)
        else:
            index = min_indexes[0]
        pred = self.id2entity[index]

        if pred == tail:
            return 1
        else:
            return 0

    def predict_triples(self, triples):
        pred=[]
        for i in triples:
            acc=self.predict(i[0], i[1], i[2])
            pred.append(acc)
        return cp.array(pred)
    def train(self):
        print("start training...")
        # Training loop

        # 将L2范数应用到实体向量矩阵上，进行归一化
        norms = cp.linalg.norm(self.entity_embeddings, axis=1)
        self.entity_embeddings = self.entity_embeddings / norms[:, cp.newaxis]
        norms = cp.linalg.norm(self.relation_embeddings, axis=1)
        self.relation_embeddings = self.relation_embeddings / norms[:, cp.newaxis]

        for epoch in range(epochs):
            print("Training epoch {}".format(epoch + 1))
            np.random.shuffle(self.train_triples)
            loss_all = 0
            # for i in range(0, len(train_triples), batch_size):
            #     batch = train_triples[i:i + batch_size]
            #     # 将训练数据转换为Cupy数组
            #     e1 = cp.array([self.entity2id[head] for head, _, _ in batch])
            #     rel = cp.array([self.relation2id[relation] for _, relation, _ in batch])
            #     e2 = cp.array([self.entity2id[tail] for _, _, tail in batch])
            #
            #     # Negative sampling
            #     neg_entities = cp.random.randint(0, len(self.entities))
            #     e2_neg = cp.array(neg_entities)
            #
            #     # 计算正样本得分
            #     pos_scores = cp.linalg.norm(
            #         self.entity_embeddings[e1] + self.relation_embeddings[rel] - self.entity_embeddings[e2], axis=1)
            #     # 计算负样本得分
            #     neg_scores = cp.linalg.norm(
            #         self.entity_embeddings[e1] + self.relation_embeddings[rel] - self.entity_embeddings[e2_neg], axis=1)
            #
            #     # 计算损失函数
            #     loss = cp.maximum(0, margin + pos_scores - neg_scores).sum()
            #     loss_all += loss
            #     # 反向传播更新参数
            #     gradient = cp.zeros_like(self.entity_embeddings)
            #     for j in range(batch_size):
            #         if pos_scores[j] - neg_scores[j] < margin:
            #             gradient[e1[j]] += (self.entity_embeddings[e1[j]] + self.relation_embeddings[rel[j]] -
            #                                 self.entity_embeddings[e2[j]])
            #             gradient[e1[j]] -= (self.entity_embeddings[e1[j]] + self.relation_embeddings[rel[j]] -
            #                                 self.entity_embeddings[e2_neg[j]])
            #     gradient *= learning_rate
            #     self.entity_embeddings -= gradient

            for i in range(0, len(self.train_triples), self.batch_size):

                batch = self.train_triples[i:i + self.batch_size]
                for head, relation, tail in batch:
                    e1 = cp.array(self.entity_embeddings[self.entity2id[head]].get())
                    rel = cp.array(self.relation_embeddings[self.relation2id[relation]].get())
                    e2 = cp.array(self.entity_embeddings[self.entity2id[tail]].get())
                    # Negative sampling
                    neg_entity = np.random.choice(list(self.entities - {head, tail}))
                    e2_neg = cp.array(self.entity_embeddings[self.entity2id[neg_entity]].get())

                    score_pos = cp.linalg.norm(e1 + rel - e2, axis=-1)
                    score_neg = cp.linalg.norm(e1 + rel - e2_neg, axis=-1)

                    loss = cp.maximum(0, self.margin + score_neg - score_pos)
                    loss_all += loss
                    gradient_pos = 2 * (e1 + rel - e2)
                    gradient_neg = 2 * (e1 + rel - e2_neg)
                    self.entity_embeddings[self.entity2id[head]] -= self.learning_rate * gradient_pos
                    self.entity_embeddings[self.entity2id[tail]] += self.learning_rate * gradient_pos
                    self.entity_embeddings[self.entity2id[neg_entity]] -= self.learning_rate * gradient_neg
                    self.relation_embeddings[self.relation2id[relation]] -= self.learning_rate * gradient_pos

            predictions_train = self.predict_triples(self.train_triples)
            accuracyt = cp.mean(predictions_train)
            predictions_valid = self.predict_triples(self.valid_triples)
            accuracy = cp.mean(predictions_valid)

            print("Epoch {}: loss = {}".format(epoch + 1, loss_all))
            print("train accuracy = {}; valid accuracy ={}".format(accuracyt, accuracy))

    # def redict_tail(self):
    #     for i in range(len(self.entity_embeddings)):
    #         head_embedding = self.entity_embeddings[i]
    #         relation_embedding = self.relation_embeddings[0]
    #         # 计算头尾实体之间的关系向量
    #         tail_embedding = head_embedding + relation_embedding
    #
    #         # 计算与关系向量最近的关系嵌入向量，即预测的关系
    #         distances = cdist([tail_embedding], self.entity_embeddings, metric='euclidean')
    #         predicted_tail_index = cp.argmin(distances)
    #         print(self.id2entity[i], self.id2entity[predicted_tail_index])


def generate_triples(num):
    entities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    relations = ['related1_to', 'related2_to', 'relation3_to', 'relation4_to', 'relation5_to', 'relation6_to']
    triples = []

    for _ in range(num):
        entity1 = random.choice(entities)
        entity2 = random.choice(entities)
        while entity1 == entity2:
            entity2 = random.choice(entities)
        triple = (entity1, random.choice(relations), entity2)
        triples.append(triple)

    # 打印生成的数据结构
    for triple in triples:
        print(triple)
    return triples


def Wn18RR2triples(file_path="../data/WN18RR/"):
    entity2id = {}
    relation2id = {}
    train_triples = []
    valid_triples = []
    test_triples = []
    with open(file_path + 'entity2id.txt') as f:
        for line in f:
            entity, entity_id = line.strip().split('\t')
            entity2id[entity] = int(entity_id)
    with open(file_path + 'relation2id.txt') as f:
        for line in f:
            relation, relation_id = line.strip().split('\t')
            relation2id[relation] = int(relation_id)

    def parse_triples(file_path, triples):
        with open(file_path) as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                triples.append((head, relation, tail))

    parse_triples(file_path + 'train.txt', train_triples)
    parse_triples(file_path + 'valid.txt', valid_triples)
    parse_triples(file_path + 'test.txt', test_triples)
    return entity2id, relation2id, train_triples, valid_triples, test_triples


if __name__ == "__main__":
    # triples=generate_triples(200)
    entity2id, relation2id, train_triples, valid_triples, test_triples = Wn18RR2triples()
    test = TransE(entity2id, relation2id, train_triples, valid_triples, test_triples)

    with open('model.pkl', 'wb') as f:
        pickle.dump(test, f)
    test.train()
    with open('model.pkl', 'wb') as f:
        pickle.dump(test, f)
    # test.predict_tail()
    # print(test.entity2id,test.relation2id)
