import torch
import numpy as np
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
nbatch = 128


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
        self.entity_embeddings = torch.rand(len(self.entities), embedding_dim, requires_grad=True)
        self.relation_embeddings = torch.rand(len(self.relations), embedding_dim,requires_grad=True)

        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        self.batch_size = int(len(train_triples) / nbatch)
        self.learning_rate = learning_rate
        self.margin = margin

    def normalization(self, vector):
        return vector / np.linalg.norm(vector)
    def train(self):
        norms = np.linalg.norm(self.entity_embeddings, axis=1)
        self.entity_embeddings = self.entity_embeddings / norms[:, np.newaxis]
        norms = np.linalg.norm(self.relation_embeddings, axis=1)
        self.relation_embeddings = self.relation_embeddings / norms[:, np.newaxis]

        train_embeddings = self.

def Wn18RR2triples(file_path="../src/WN18RR/"):
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    entity2id, relation2id, train_triples, valid_triples, test_triples = Wn18RR2triples()
    print(type(entity2id), type(relation2id), type(train_triples), type(valid_triples), type(test_triples))
    # test = TransE(entity2id, relation2id, train_triples, valid_triples, test_triples)
    #
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(test, f)
    # test.train()
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(test, f)
    # test.predict_tail()
    # print(test.entity2id,test.relation2id)
