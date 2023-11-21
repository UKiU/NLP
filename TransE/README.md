# 安装cuda+pytorch
https://blog.csdn.net/CDL_LuFei/article/details/124012894
# 代码说明
## TransEcode cpu版本的代码

## GPUtest cuda+cupy版本的代码
实测不如上面那个

## torch_test cuda+pytorch版本
参考https://zhuanlan.zhihu.com/p/508508180

训练后保存实体向量、关系向量、训练损失记录和图

新增基于启发式采样分布（tph\hpt）的负采样算法

使用Xavier初始化替代原文的初始化方法

使用SGD或者Adam优化器进行优化

    # Hyperparameters

    learning_rate: 学习率
    embedding_dim：实体、关系维度
    margin：正例、负例边界
    epochs：训练轮次
    batch_size：训练批次大小
    
    Wn18RR2triples（）：读取数据集
    entity_list,relation_list：entity 和 relation 的 id 的列表
    entity2id, relation2id：{name：id} 的字典 
    train_triples, valid_triples, test_triples：[[h_id,r_id,t_id]] 的列表
    relation_tph, relation_hpt: 每个 relation 的平均尾节点数 hpt 和平均头结点数 tph
    
    TransE 参数：
    entity_list,relation_list,entity2id, relation2id, train_triples, valid_triples, test_triples, relation_tph, relation_hpt
    TransE.training_run(out_file_title="WN18RR_torch_")