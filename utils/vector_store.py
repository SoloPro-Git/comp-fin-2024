import numpy as np
import json
from rank_bm25 import BM25Okapi
import jieba
import requests
import os
import pickle
import faiss
from tqdm import tqdm

class VectorStore:
    def __init__(self):
        self.api_key = self._load_api_key()
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        self.rerank_url = "https://api.siliconflow.cn/v1/rerank"
        self.documents = []
        self.embeddings = None
        self.bm25 = None
        self.vector_store_dir = 'vector_store'
        self.index = None
        self.embedding_size = 1024  # BGE-large-zh 的向量维度
        self.load_data()
    
    def _load_api_key(self):
        """从apikey文件加载API密钥"""
        try:
            with open('apikey', 'r') as f:
                for line in f:
                    if line.startswith('SILICON_FLOW_API_KEY='):
                        return line.strip().split('=')[1]
            raise ValueError("未找到SILICON_FLOW_API_KEY配置")
        except FileNotFoundError:
            raise FileNotFoundError("未找到apikey文件，请在根目录创建apikey文件")
    
    def _ensure_vector_store_dir(self):
        """确保向量存储目录存在"""
        if not os.path.exists(self.vector_store_dir):
            os.makedirs(self.vector_store_dir)
    
    def _init_faiss_index(self):
        """初始化FAISS索引"""
        quantizer = faiss.IndexFlatIP(self.embedding_size)
        nlist = 100  # 聚类中心数量，可以根据数据规模调整
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_size, nlist, faiss.METRIC_INNER_PRODUCT)
    
    def _normalize_vectors(self, vectors):
        """归一化向量，用于余弦相似度计算"""
        return vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    def get_embedding(self, text):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "BAAI/bge-large-zh-v1.5",
            "input": text,
            "encoding_format": "float"
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            raise Exception(f"API调用失败: {response.text}")
    
    def _save_vector_store(self, documents, embeddings, tokenized_documents):
        """保存向量存储到本地"""
        self._ensure_vector_store_dir()
        
        # 保存文档和分词结果
        with open(os.path.join(self.vector_store_dir, 'documents.pkl'), 'wb') as f:
            pickle.dump({'documents': documents, 'tokenized': tokenized_documents}, f)
        
        # 归一化向量
        normalized_vectors = self._normalize_vectors(embeddings)
        
        # 初始化和训练FAISS索引
        if self.index is None:
            self._init_faiss_index()
        
        if not self.index.is_trained:
            self.index.train(normalized_vectors)
        
        # 添加向量到索引
        self.index.add(normalized_vectors)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(self.vector_store_dir, 'faiss.index'))
    
    def _load_vector_store(self):
        """从本地加载向量存储"""
        try:
            # 加载文档和分词结果
            with open(os.path.join(self.vector_store_dir, 'documents.pkl'), 'rb') as f:
                data = pickle.load(f)
                documents = data['documents']
                tokenized_documents = data['tokenized']
            
            # 加载FAISS索引
            self.index = faiss.read_index(os.path.join(self.vector_store_dir, 'faiss.index'))
            
            return documents, tokenized_documents
        except (FileNotFoundError, EOFError):
            return None, None
    
    def load_data(self):
        """加载数据，优先从本地向量存储加载"""
        # 尝试从本地加载
        documents, tokenized_documents = self._load_vector_store()
        
        if documents is not None and self.index is not None:
            print("从本地向量存储加载数据...")
            self.documents = documents
            self.bm25 = BM25Okapi(tokenized_documents)
            return
        
        print("从原始数据构建向量存储...")
        # 从JSON文件加载数据
        with open('data_graph/output/graph_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 提取所有节点的文本内容
        for node in data['nodes']:
            self.documents.append(str(node))
        
        # 计算所有文档的向量表示
        print("计算文档向量...")
        embeddings = []
        for doc in tqdm(self.documents, desc="向量化进度"):
            embedding = self.get_embedding(doc)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        # 准备BM25
        print("构建BM25索引...")
        tokenized_documents = [list(jieba.cut(doc)) for doc in tqdm(self.documents, desc="分词进度")]
        self.bm25 = BM25Okapi(tokenized_documents)
        
        # 保存到本地
        print("保存向量存储到本地...")
        self._save_vector_store(self.documents, embeddings, tokenized_documents)
        print("向量存储构建完成！")
    
    def rerank_results(self, query, documents, top_k=5):
        """使用重排序API对结果进行重新排序"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": query,
            "documents": documents,
            "top_n": top_k,
            "return_documents": True
        }
        
        response = requests.post(self.rerank_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['results']
        else:
            raise Exception(f"重排序API调用失败: {response.text}")
    
    def hybrid_search(self, query, top_k=5, rerank_candidates=20):
        """混合搜索并使用重排序优化结果"""
        # 获取查询文本的向量
        query_vector = self.get_embedding(query)
        
        # 归一化查询向量并使用FAISS搜索
        normalized_query = self._normalize_vectors(query_vector.reshape(1, -1))
        scores, indices = self.index.search(normalized_query, rerank_candidates)
        
        # 获取相似度分数和候选索引
        cosine_scores = scores[0]  # 取第一行，因为只有一个查询
        candidate_indices = indices[0]
        
        # 计算BM25分数
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = bm25_scores[candidate_indices]  # 只取候选文档的分数
        
        # 归一化BM25分数
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        
        # 组合分数
        combined_scores = 0.7 * cosine_scores + 0.3 * bm25_scores
        
        # 准备重排序的文档
        candidate_docs = [self.documents[idx] for idx in candidate_indices]
        
        # 使用重排序API优化结果
        reranked_results = self.rerank_results(query, candidate_docs, top_k)
        
        # 格式化最终结果
        results = []
        for result in reranked_results:
            idx = candidate_indices[result['index']]
            results.append({
                'text': result['document']['text'],
                'score': float(result['relevance_score']),
                'vector_score': float(cosine_scores[result['index']]),
                'bm25_score': float(bm25_scores[result['index']])
            })
        
        return results