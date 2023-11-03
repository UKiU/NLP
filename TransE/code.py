import numpy as np
from scipy.spatial.distance import cdist
import random
# Hyperparameters
learning_rate = 0.01
embedding_dim = 50
margin = 1.0
epochs = 100
batch_size = 128


class TransE:
    def __init__(self, triples):
        self.triples = triples
        # Extract unique entities and relations
        self.entities = list(set([triple[0] for triple in triples] + [triple[2] for triple in triples]))
        self.relations = list(set([triple[1] for triple in triples]))

        # Create dictionaries to map entities and relations to unique IDs
        self.entity2id = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relation2id = {relation: idx for idx, relation in enumerate(self.relations)}

        self.id2entity = {idx: entity for idx, entity in enumerate(self.entities)}
        self.id2relation = {idx: relation for idx, relation in enumerate(self.relations)}

        # Initialize entity and relation embeddings randomly
        self.entity_embeddings = np.random.rand(len(self.entities), embedding_dim)
        self.relation_embeddings = np.random.rand(len(self.relations), embedding_dim)

    def train(self):

        # Training loop
        for epoch in range(epochs):
            np.random.shuffle(self.triples)
            total_loss = 0

            for i in range(0, len(self.triples), batch_size):
                batch = self.triples[i:i + batch_size]
                pos_heads = [self.entity2id[triple[0]] for triple in batch]
                pos_relations = [self.relation2id[triple[1]] for triple in batch]
                pos_tails = [self.entity2id[triple[2]] for triple in batch]

                # Negative sampling
                neg_heads = np.random.choice(len(self.entities), len(batch))
                neg_relations = np.random.choice(len(self.relations), len(batch))
                neg_tails = np.random.choice(len(self.entities), len(batch))

                # Calculate the score for positive and negative triples
                pos_scores = np.linalg.norm(
                    self.entity_embeddings[pos_heads] + self.relation_embeddings[pos_relations] -
                    self.entity_embeddings[pos_tails], axis=1)
                neg_scores = np.linalg.norm(
                    self.entity_embeddings[neg_heads] + self.relation_embeddings[neg_relations] -
                    self.entity_embeddings[neg_tails], axis=1)

                # Compute the margin loss
                loss = np.maximum(0, margin + pos_scores - neg_scores)
                total_loss += np.sum(loss)

                # Compute the gradients and update embeddings
                for j in range(len(batch)):
                    if loss[j] > 0:
                        grad_pos_head = 2 * (
                                self.entity_embeddings[pos_heads[j]] + self.relation_embeddings[pos_relations[j]] -
                                self.entity_embeddings[pos_tails[j]])
                        grad_pos_tail = -2 * (
                                self.entity_embeddings[pos_heads[j]] + self.relation_embeddings[pos_relations[j]] -
                                self.entity_embeddings[pos_tails[j]])
                        grad_neg_head = 2 * (
                                self.entity_embeddings[neg_heads[j]] + self.relation_embeddings[neg_relations[j]] -
                                self.entity_embeddings[neg_tails[j]])
                        grad_neg_tail = -2 * (
                                self.entity_embeddings[neg_heads[j]] + self.relation_embeddings[neg_relations[j]] -
                                self.entity_embeddings[neg_tails[j]])

                        self.entity_embeddings[pos_heads[j]] -= learning_rate * grad_pos_head
                        self.entity_embeddings[pos_tails[j]] -= learning_rate * grad_pos_tail
                        self.entity_embeddings[neg_heads[j]] -= learning_rate * grad_neg_head
                        self.entity_embeddings[neg_tails[j]] -= learning_rate * grad_neg_tail
                        self.relation_embeddings[pos_relations[j]] -= learning_rate * grad_pos_head
                        self.relation_embeddings[neg_relations[j]] -= learning_rate * grad_neg_head

            print(f"Epoch {epoch + 1}, Loss: {total_loss}")

        # Save the learned embeddings for future use
        np.save("entity_embeddings.npy", self.entity_embeddings)
        np.save("relation_embeddings.npy", self.relation_embeddings)

    def predict_tail(self):
        for i in range(len(self.entity_embeddings)):
            head_embedding = self.entity_embeddings[i]
            relation_embedding = self.relation_embeddings[0]
            # 计算头尾实体之间的关系向量
            tail_embedding = head_embedding + relation_embedding

            # 计算与关系向量最近的关系嵌入向量，即预测的关系
            distances = cdist([tail_embedding], self.entity_embeddings, metric='euclidean')
            predicted_tail_index = np.argmin(distances)
            print(self.id2entity[i], self.id2entity[predicted_tail_index])

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

if __name__ == "__main__":
    # Data preparation: You need to have a knowledge graph in the form of triples (head, relation, tail)
    # In this example, I'll use a small toy knowledge graph

    # 随机生成函数优点问题，先别用
    # triples=generate_triples(200)
    triples = [
        ("A", "related_to", "B"),
        ("B", "related_to", "C"),
        ("D", "related_to", "C"),
        ("E", "related_to", "F"),
    ]

    test = TransE(triples)
    test.train()
    test.predict_tail()
    print(test.entity2id,test.relation2id)