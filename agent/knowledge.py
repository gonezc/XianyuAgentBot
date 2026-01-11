"""
知识库（RAG增强版）

支持两种检索模式：
1. FAISS 向量检索（推荐，需安装 faiss-cpu）
2. 关键词匹配（降级方案）
"""
import os
import json
import pickle
from typing import List, Dict, Optional
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 尝试导入 FAISS
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS 未安装，使用关键词匹配")

# 尝试导入 OpenAI（用于 Embedding）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class KnowledgeBase:
    """知识库（支持向量检索）"""
    
    def __init__(self, knowledge_dir: str = None):
        if knowledge_dir is None:
            knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge")
        
        self.knowledge_dir = knowledge_dir
        self.cases: List[Dict] = []
        self.skills: str = ""
        self.methods: str = ""
        
        # FAISS 相关
        self.index: Optional[faiss.IndexFlatIP] = None if FAISS_AVAILABLE else None
        self.embeddings_cache: Dict[str, List[float]] = {}
        self.embedding_dim = 1024  # text-embedding-v3 支持: 64/128/256/512/768/1024
        
        # 索引文件路径
        self.index_path = os.path.join(knowledge_dir, ".faiss_index")
        self.cache_path = os.path.join(knowledge_dir, ".embeddings_cache.pkl")
        
        self._load()
    
    def _get_embedding_client(self) -> Optional[OpenAI]:
        """获取 Embedding 客户端"""
        if not OPENAI_AVAILABLE:
            return None
        
        api_key = os.getenv("API_KEY")
        if not api_key:
            return None
        
        return OpenAI(
            api_key=api_key,
            base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的向量表示"""
        # 检查缓存
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        client = self._get_embedding_client()
        if not client:
            return None
        
        try:
            response = client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
                input=text,
                dimensions=self.embedding_dim
            )
            embedding = response.data[0].embedding
            self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"获取 Embedding 失败: {e}")
            return None
    
    def _build_index(self):
        """构建 FAISS 索引"""
        if not FAISS_AVAILABLE or not self.cases:
            return
        
        logger.info("构建 FAISS 索引...")
        
        # 为每个案例生成向量
        texts = []
        valid_cases = []
        
        for case in self.cases:
            # 组合搜索文本
            text = f"{case.get('title', '')} {case.get('description', '')} {' '.join(case.get('tags', []))}"
            embedding = self._get_embedding(text)
            
            if embedding:
                texts.append(embedding)
                valid_cases.append(case)
        
        if not texts:
            logger.warning("无法生成向量，FAISS 索引构建失败")
            return
        
        # 创建索引（内积相似度）
        vectors = np.array(texts, dtype=np.float32)
        # 归一化（用于余弦相似度）
        faiss.normalize_L2(vectors)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(vectors)
        self.cases = valid_cases  # 只保留有向量的案例
        
        # 保存索引和缓存
        self._save_index()
        
        logger.info(f"FAISS 索引构建完成，共 {len(valid_cases)} 条案例")
    
    def _save_index(self):
        """保存索引到文件"""
        if self.index is None:
            return
        
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.embeddings_cache, f)
            logger.debug("FAISS 索引已保存")
        except Exception as e:
            logger.warning(f"保存索引失败: {e}")
    
    def _load_index(self) -> bool:
        """加载已有索引"""
        if not FAISS_AVAILABLE:
            return False
        
        if not os.path.exists(self.index_path):
            return False
        
        try:
            self.index = faiss.read_index(self.index_path)
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    self.embeddings_cache = pickle.load(f)
            logger.info(f"加载已有 FAISS 索引，共 {self.index.ntotal} 条")
            return True
        except Exception as e:
            logger.warning(f"加载索引失败: {e}")
            return False
    
    def _load(self):
        """加载知识库"""
        # 案例库
        cases_path = os.path.join(self.knowledge_dir, "cases.json")
        if os.path.exists(cases_path):
            try:
                with open(cases_path, "r", encoding="utf-8") as f:
                    self.cases = json.load(f)
                logger.info(f"加载案例库: {len(self.cases)} 条")
            except Exception as e:
                logger.warning(f"加载案例库失败: {e}")
        
        # 技能库
        skills_path = os.path.join(self.knowledge_dir, "skills.json")
        if os.path.exists(skills_path):
            try:
                with open(skills_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.skills = "\n".join(
                    f"- {s['name']}({s.get('level', '')})：{s.get('description', '')}"
                    for s in data
                )
            except Exception as e:
                logger.warning(f"加载技能库失败: {e}")
        
        # 方法库
        methods_path = os.path.join(self.knowledge_dir, "methods.json")
        if os.path.exists(methods_path):
            try:
                with open(methods_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.methods = "\n".join(
                    f"- {m['name']}：适用于{', '.join(m.get('scenarios', []))}"
                    for m in data
                )
            except Exception as e:
                logger.warning(f"加载方法库失败: {e}")
        
        # 尝试加载或构建 FAISS 索引
        if FAISS_AVAILABLE and self.cases:
            if not self._load_index():
                self._build_index()
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相似案例
        
        优先使用 FAISS 向量检索，降级使用关键词匹配
        """
        if not self.cases:
            return []
        
        # 查询预处理：常见拼音转中文
        query = self._normalize_query(query)
        
        # 尝试向量检索
        if self.index is not None:
            return self._search_faiss(query, top_k)
        
        # 降级：关键词匹配
        return self._search_keyword(query, top_k)
    
    def _normalize_query(self, query: str) -> str:
        """查询预处理：拼音转中文、同义词替换等"""
        # 常见拼音到中文的映射
        pinyin_map = {
            'gongdan': '工单',
            'xitong': '系统',
            'quanxian': '权限',
            'liuzhuan': '流转',
            'tushu': '图书',
            'baoxiu': '报修',
            'sushe': '宿舍',
        }
        
        query_lower = query.lower()
        for pinyin, chinese in pinyin_map.items():
            if pinyin in query_lower:
                # 替换拼音为中文（保持原大小写风格）
                query = query.replace(pinyin, chinese).replace(pinyin.capitalize(), chinese)
        
        return query
    
    def _extract_keywords(self, text: str) -> set:
        """提取查询中的关键词（去除停用词）"""
        import re
        # 中文停用词
        stopwords = {'的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一', '一个', 
                     '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', 
                     '好', '自己', '这', '做', '能', '吗', '什么', '怎么', '多少', '可以', '要'}
        
        # 提取中文词（2字以上）和英文单词
        words = set()
        # 中文词（2-4字）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
        words.update([w for w in chinese_words if w not in stopwords])
        # 英文单词（3字符以上）
        english_words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        words.update(english_words)
        
        return words
    
    def _check_keyword_match(self, query: str, case: Dict) -> float:
        """检查关键词匹配度，返回0-1的分数
        
        如果查询中的关键词在案例中完全找不到，返回0.3（不完全过滤，但降低权重）
        如果部分匹配，返回0.5-0.9
        如果完全匹配，返回1.0
        """
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return 1.0  # 没有关键词，不进行过滤
        
        # 合并案例的title、tags、description
        case_text = f"{case.get('title', '')} {' '.join(case.get('tags', []))} {case.get('description', '')[:200]}"
        case_text_lower = case_text.lower()
        
        matched = 0
        for keyword in query_keywords:
            if keyword in case_text_lower or keyword.lower() in case_text_lower:
                matched += 1
        
        if matched == 0:
            return 0.3  # 完全无匹配，但不完全过滤，只是降低权重
        elif matched == len(query_keywords):
            return 1.0  # 完全匹配
        else:
            return 0.5 + 0.4 * (matched / len(query_keywords))  # 部分匹配

    def _search_faiss(self, query: str, top_k: int, min_score: float = 0.6) -> List[Dict]:
        """FAISS 向量检索

        Args:
            query: 搜索关键词
            top_k: 返回结果数量
            min_score: 最低相似度阈值，默认0.6
        """
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.warning("查询向量化失败，降级为关键词匹配")
            return self._search_keyword(query, top_k)

        # 归一化查询向量
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # 搜索候选
        candidate_count = min(top_k * 10, len(self.cases))
        scores, indices = self.index.search(query_vec, candidate_count)
        scores = scores[0]

        # 收集结果
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.cases):
                semantic_score = float(scores[i])
                if semantic_score < min_score:
                    continue

                case = self.cases[idx].copy()
                keyword_score = self._check_keyword_match(query, case)

                # 混合评分
                keyword_boost = 0.2 * keyword_score
                rank_boost = 0.1 * (1.0 - i / candidate_count) if i < top_k else 0
                final_score = semantic_score * (0.7 + keyword_boost + rank_boost)

                case["_score"] = final_score
                case["_semantic_score"] = semantic_score
                case["_keyword_score"] = keyword_score
                candidates.append(case)
        
        # 按综合分数排序
        candidates.sort(key=lambda x: x["_score"], reverse=True)
        
        # 如果关键词完全不匹配的结果太多，进行二次过滤
        # 但保留至少top_k个结果（即使关键词匹配度低）
        results = []
        keyword_matched = []
        keyword_unmatched = []
        
        for case in candidates:
            if case["_keyword_score"] >= 0.5:  # 关键词匹配度>=0.5
                keyword_matched.append(case)
            else:
                keyword_unmatched.append(case)
        
        # 优先使用关键词匹配的结果
        results.extend(keyword_matched[:top_k])
        
        # 如果关键词匹配的结果不够，补充语义相似但关键词不匹配的结果
        # 但要求语义分数足够高（>0.55）且与Top1差距不太大
        if len(results) < top_k and keyword_matched:
            top1_semantic = keyword_matched[0]["_semantic_score"] if keyword_matched else 0
            for case in keyword_unmatched:
                if len(results) >= top_k:
                    break
                # 语义分数要>0.55，且与Top1差距<0.15
                if case["_semantic_score"] > 0.55 and (top1_semantic - case["_semantic_score"]) < 0.15:
                    results.append(case)
        elif len(results) < top_k:
            # 如果完全没有关键词匹配的结果，直接使用语义相似度最高的
            results.extend(candidates[:top_k])
        
        results = results[:top_k]  # 确保不超过top_k
        
        logger.info(f"FAISS 搜索 '{query[:20]}...' 返回 {len(results)} 条 (自适应阈值={min_score:.3f}, 关键词匹配={len(keyword_matched)})")
        return results
    
    def _search_keyword(self, query: str, top_k: int) -> List[Dict]:
        """关键词匹配检索（降级方案）"""
        scored = []
        query_lower = query.lower()
        
        for case in self.cases:
            score = 0
            title = case.get("title", "").lower()
            desc = case.get("description", "").lower()
            tags = " ".join(case.get("tags", [])).lower()
            
            for word in query_lower.split():
                if len(word) < 2:
                    continue
                if word in title:
                    score += 3
                if word in desc:
                    score += 2
                if word in tags:
                    score += 1
            
            if score > 0:
                case_copy = case.copy()
                case_copy["_score"] = score
                scored.append((score, case_copy))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [c for _, c in scored[:top_k]]
        
        logger.info(f"关键词搜索 '{query[:20]}...' 返回 {len(results)} 条")
        return results
    
    def rebuild_index(self):
        """重建索引（知识库更新后调用）"""
        if FAISS_AVAILABLE:
            # 清除旧索引
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)
            self.embeddings_cache = {}
            self.index = None
            
            # 重新构建
            self._build_index()
