from utils.vector_store import VectorStore

def main():
    # 初始化向量存储
    vector_store = VectorStore()
    # vector_store.rebuild_bm25_index()
    
    # 测试查询
    query = "600872的全称、A股简称、法人、法律顾问、会计师事务所及董秘是？"
    results = vector_store.hybrid_search(query, top_k=5, rerank_candidates=20)
    
    # 打印结果
    print(f"\n查询: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"文本: {result['text']}")
        print(f"综合得分: {result['score']:.4f}")
        print(f"向量相似度: {result['vector_score']:.4f}")
        print(f"BM25得分: {result['bm25_score']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 