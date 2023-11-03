import numpy as np

# Hyperparameters
learning_rate = 0.01
embedding_dim = 50
margin = 1.0
epochs = 100
batch_size = 128

# Data preparation: You need to have a knowledge graph in the form of triples (head, relation, tail)
# In this example, I'll use a small toy knowledge graph
triples = [
    ("A", "related_to", "B"),
    ("B", "related_to", "C"),
    ("D", "related_to", "C"),
    ("E", "related_to", "F"),
]

# Extract unique entities and relations
entities = list(set([triple[0] for triple in triples] + [triple[2] for triple in triples]))
relations = list(set([triple[1] for triple in triples]))

# Create dictionaries to map entities and relations to unique IDs
entity2id = {entity: idx for idx, entity in enumerate(entities)}
relation2id = {relation: idx for idx, relation in enumerate(relations)}

# Initialize entity and relation embeddings randomly
entity_embeddings = np.random.rand(len(entities), embedding_dim)
relation_embeddings = np.random.rand(len(relations), embedding_dim)

# Training loop
for epoch in range(epochs):
    np.random.shuffle(triples)
    total_loss = 0

    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        pos_heads = [entity2id[triple[0]] for triple in batch]
        pos_relations = [relation2id[triple[1]] for triple in batch]
        pos_tails = [entity2id[triple[2]] for triple in batch]

        # Negative sampling
        neg_heads = np.random.choice(len(entities), len(batch))
        neg_relations = np.random.choice(len(relations), len(batch))
        neg_tails = np.random.choice(len(entities), len(batch))

        # Calculate the score for positive and negative triples
        pos_scores = np.linalg.norm(entity_embeddings[pos_heads] + relation_embeddings[pos_relations] - entity_embeddings[pos_tails], axis=1)
        neg_scores = np.linalg.norm(entity_embeddings[neg_heads] + relation_embeddings[neg_relations] - entity_embeddings[neg_tails], axis=1)

        # Compute the margin loss
        loss = np.maximum(0, margin + pos_scores - neg_scores)
        total_loss += np.sum(loss)

        # Compute the gradients and update embeddings
        for j in range(len(batch)):
            if loss[j] > 0:
                grad_pos_head = 2 * (entity_embeddings[pos_heads[j]] + relation_embeddings[pos_relations[j]] - entity_embeddings[pos_tails[j]])
                grad_pos_tail = -2 * (entity_embeddings[pos_heads[j]] + relation_embeddings[pos_relations[j]] - entity_embeddings[pos_tails[j])
                grad_neg_head = 2 * (entity_embeddings[neg_heads[j]] + relation_embeddings[neg_relations[j]] - entity_embeddings[neg_tails[j]])
                grad_neg_tail = -2 * (entity_embeddings[neg_heads[j]] + relation_embeddings[neg_relations[j]] - entity_embeddings[neg_tails[j])

                entity_embeddings[pos_heads[j]] -= learning_rate * grad_pos_head
                entity_embeddings[pos_tails[j]] -= learning_rate * grad_pos_tail
                entity_embeddings[neg_heads[j]] -= learning_rate * grad_neg_head
                entity_embeddings[neg_tails[j]] -= learning_rate * grad_neg_tail
                relation_embeddings[pos_relations[j]] -= learning_rate * grad_pos_head
                relation_embeddings[neg_relations[j]] -= learning_rate * grad_neg_head

    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Save the learned embeddings for future use
np.save("entity_embeddings.npy", entity_embeddings)
np.save("relation_embeddings.npy", relation_embeddings)
