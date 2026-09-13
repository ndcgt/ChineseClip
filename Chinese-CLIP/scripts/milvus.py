from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np


# 连接到 Milvus 服务器
def connect_to_milvus(host, port):
    connections.connect(host=host, port=port)


# 创建 Milvus 集合
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, description='ids', is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
        FieldSchema(name='category', dtype=DataType.INT64, description='category'),
    ]

    schema = CollectionSchema(fields=fields, description='Image and text embeddings')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }

    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


# 插入数据到 Milvus 集合
def insert_data_to_milvus(collection, ids, embeddings, categories):
    data = [ids, embeddings, categories]
    mr = collection.insert(data)
    return mr


if __name__ == "__main__":
    # Milvus 服务器配置
    milvus_host = "localhost"
    milvus_port = "19530"
    collection_name = "image_text_embeddings"
    vector_dim = 512

    # 连接到 Milvus 服务器
    connect_to_milvus(milvus_host, milvus_port)

    # 创建集合
    collection = create_milvus_collection(collection_name, vector_dim)

    # 示例嵌入向量数据
    # num_entities = 10
    # ids = list(range(num_entities))
    # embeddings = np.random.rand(num_entities, vector_dim).astype(np.float32)
    # categories = [1] * num_entities

    # 插入数据到 Milvus
    insert_result = insert_data_to_milvus(collection, ids, embeddings, categories)
    print(f"Inserted {insert_result.insert_count} entities into Milvus.")

    # 加载集合到内存
    collection.load()

    # 获取集合中的实体数量
    num_entities_in_collection = collection.num_entities
    print(f"Number of entities in the collection: {num_entities_in_collection}")
