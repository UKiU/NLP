from torch_geometric.datasets import KarateClub

class Data(object):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, **kwargs):
        r"""
        Args:
            x (Tensor, optional): 节点属性矩阵，大小为`[num_nodes, num_node_features]`
            edge_index (LongTensor, optional): 边索引矩阵，大小为`[2, num_edges]`，第0行为尾节点，第1行为头节点，头指向尾
            edge_attr (Tensor, optional): 边属性矩阵，大小为`[num_edges, num_edge_features]`
            y (Tensor, optional): 节点或图的标签，任意大小（，其实也可以是边的标签）
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y

        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item


dataset = KarateClub()
data = dataset[0]  # Get the first graph object.
print(data)
print('==============================================================')

# 获取图的一些信息
print(f'Number of nodes: {data.num_nodes}') # 节点数量
print(f'Number of edges: {data.num_edges}') # 边数量
print(f'Number of node features: {data.num_node_features}') # 节点属性的维度
print(f'Number of node features: {data.num_features}') # 同样是节点属性的维度
print(f'Number of edge features: {data.num_edge_features}') # 边属性的维度
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # 平均节点度
print(f'if edge indices are ordered and do not contain duplicate entries.: {data.is_coalesced()}') # 是否边是有序的同时不含有重复的边
print(f'Number of training nodes: {data.train_mask.sum()}') # 用作训练集的节点
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}') # 用作训练集的节点的数量
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}') # 此图是否包含孤立的节点
print(f'Contains self-loops: {data.contains_self_loops()}')  # 此图是否包含自环的边
print(f'Is undirected: {data.is_undirected()}')  # 此图是否是无向图