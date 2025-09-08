# MediAgents-LangGraph 架构优化指南

基于九大步骤框架的 AI Agent 系统升级方案

## 📋 项目现状分析

### 当前架构特点
- **简单流程架构**: 基于难度分级的线性处理（basic → intermediate → advanced）
- **单一模型依赖**: 依赖 Gemini/OpenAI 单一模型处理所有任务
- **有限协作机制**: 基础的 Agent 类和 Group 类，缺乏复杂交互
- **成本控制导向**: 强调 token 使用效率，但限制了功能扩展

### 核心痛点识别
1. **架构僵化**: 缺乏灵活的工作流编排能力
2. **记忆缺失**: 无长期记忆和上下文管理
3. **工具受限**: 缺乏外部工具集成和 RAG 能力
4. **单一交互**: 仅支持命令行，缺乏前端界面
5. **评估单一**: 仅基于准确性，缺乏多维度评估

---

## 🎯 九大步骤优化方案

### Step 1: 明确目标与场景重新定义

#### 优化目标
- **从单一问答** → **多模态医疗助手**
- **从成本优先** → **性能与成本平衡**
- **从评估工具** → **生产就绪的医疗 AI 系统**

#### 核心场景扩展
```python
# 新增应用场景
SCENARIOS = {
    "diagnostic_support": "诊断支持和建议",
    "treatment_planning": "治疗方案制定", 
    "literature_review": "文献综述和研究",
    "medical_education": "医学教育和培训",
    "clinical_decision": "临床决策支持"
}
```

#### 用户群体
- **医学生**: 学习辅助和考试准备
- **临床医生**: 诊断支持和决策辅助  
- **研究人员**: 文献分析和研究支持
- **患者**: 健康咨询和信息获取

### Step 2: 规范输入输出架构

#### 输入标准化
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class QueryType(str, Enum):
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment" 
    RESEARCH = "research"
    EDUCATION = "education"

class MedicalQuery(BaseModel):
    query_text: str = Field(..., min_length=10, max_length=2000)
    query_type: QueryType
    context: Optional[Dict[str, Any]] = None
    patient_info: Optional[Dict[str, str]] = None
    priority: int = Field(default=1, ge=1, le=5)
    require_sources: bool = Field(default=True)
    max_response_length: int = Field(default=500, ge=100, le=1000)

class MedicalContext(BaseModel):
    symptoms: Optional[List[str]] = None
    medical_history: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    test_results: Optional[Dict[str, Any]] = None
```

#### 输出标准化
```python
class MedicalResponse(BaseModel):
    response_id: str
    query_id: str
    primary_response: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    supporting_evidence: List[Dict[str, str]]
    alternative_perspectives: Optional[List[str]] = None
    follow_up_questions: Optional[List[str]] = None
    sources: List[Dict[str, str]]
    processing_metadata: ProcessingMetadata

class ProcessingMetadata(BaseModel):
    agents_involved: List[str]
    processing_time: float
    difficulty_level: str
    token_usage: TokenUsage
    workflow_path: List[str]
```

### Step 3: 基于 LangGraph 的提示词工程

#### Agent 角色定义
```python
AGENT_PROMPTS = {
    "diagnostic_specialist": {
        "role": "你是一位经验丰富的诊断专家",
        "personality": "严谨、细致、基于循证医学",
        "specialization": "症状分析、鉴别诊断、诊断建议",
        "output_format": "结构化诊断报告"
    },
    
    "treatment_planner": {
        "role": "你是一位治疗方案专家", 
        "personality": "实用、全面、考虑患者个体差异",
        "specialization": "治疗方案制定、药物选择、疗效评估",
        "output_format": "分层治疗建议"
    },
    
    "research_analyst": {
        "role": "你是一位医学研究分析师",
        "personality": "客观、批判性思维、循证导向",
        "specialization": "文献检索、证据评估、研究综述",
        "output_format": "循证医学报告"
    }
}
```

#### 动态提示词生成
```python
def generate_context_aware_prompt(agent_type: str, query: MedicalQuery, context: Dict) -> str:
    base_prompt = AGENT_PROMPTS[agent_type]
    
    # 根据查询类型和上下文动态调整
    if query.query_type == QueryType.DIAGNOSIS:
        return f"""
        {base_prompt['role']}，{base_prompt['personality']}。
        
        患者咨询: {query.query_text}
        
        已知信息:
        - 症状: {context.get('symptoms', '未提供')}
        - 病史: {context.get('medical_history', '未提供')}
        
        请提供:
        1. 可能诊断（按概率排序）
        2. 进一步检查建议
        3. 鉴别诊断要点
        
        输出格式: {base_prompt['output_format']}
        字数限制: {query.max_response_length} 字以内
        """
```

### Step 4: LangGraph 工作流与工具集成

#### 核心工作流设计
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class MedicalWorkflowState(TypedDict):
    query: MedicalQuery
    context: MedicalContext
    difficulty_assessment: str
    agent_responses: Annotated[List[Dict], operator.add]
    final_synthesis: str
    confidence_scores: Annotated[List[float], operator.add]
    sources: Annotated[List[Dict], operator.add]

def create_medical_workflow() -> StateGraph:
    workflow = StateGraph(MedicalWorkflowState)
    
    # 节点定义
    workflow.add_node("difficulty_assessor", assess_query_difficulty)
    workflow.add_node("rag_retriever", retrieve_relevant_knowledge)
    workflow.add_node("specialist_panel", convene_specialist_panel)
    workflow.add_node("evidence_evaluator", evaluate_evidence_quality)
    workflow.add_node("response_synthesizer", synthesize_final_response)
    workflow.add_node("quality_checker", perform_quality_check)
    
    # 条件路由
    workflow.add_conditional_edges(
        "difficulty_assessor",
        route_by_difficulty,
        {
            "basic": "specialist_panel",
            "intermediate": "rag_retriever", 
            "advanced": "evidence_evaluator"
        }
    )
    
    return workflow.compile()
```

#### RAG 工具集成
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PubMedLoader

class MedicalRAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name="medical_knowledge",
            embedding_function=self.embeddings
        )
        self.pubmed_loader = PubMedLoader()
    
    async def retrieve_context(self, query: str, k: int = 5) -> List[Document]:
        # 混合检索：向量相似度 + 关键词匹配
        vector_results = self.vectorstore.similarity_search(query, k=k)
        
        # 实时 PubMed 检索
        recent_papers = await self.pubmed_loader.load_recent_papers(
            query=query, 
            days_back=30,
            max_results=3
        )
        
        return vector_results + recent_papers
```

### Step 5: 多 Agent 协作架构

#### 专业化 Agent 设计
```python
class SpecializedMedicalAgent:
    def __init__(self, specialty: str, model_config: dict):
        self.specialty = specialty
        self.model = self.initialize_model(model_config)
        self.memory = ConversationBufferMemory()
        self.tools = self.get_specialty_tools()
    
    def get_specialty_tools(self) -> List[Tool]:
        specialty_tools = {
            "diagnostic": [
                DifferentialDiagnosisTool(),
                SymptomAnalysisTool(),
                TestRecommendationTool()
            ],
            "treatment": [
                DrugInteractionTool(), 
                GuidelineSearchTool(),
                DosageCalculatorTool()
            ],
            "research": [
                PubMedSearchTool(),
                ClinicalTrialsTool(),
                EvidenceGradingTool()
            ]
        }
        return specialty_tools.get(self.specialty, [])

class MedicalTeamCoordinator:
    def __init__(self):
        self.agents = {
            "diagnostician": SpecializedMedicalAgent("diagnostic", GEMINI_CONFIG),
            "clinician": SpecializedMedicalAgent("treatment", GEMINI_CONFIG), 
            "researcher": SpecializedMedicalAgent("research", GEMINI_CONFIG),
            "coordinator": CoordinatorAgent(GPT4_CONFIG)
        }
    
    async def process_query(self, query: MedicalQuery) -> MedicalResponse:
        # 并行专家咨询
        expert_responses = await asyncio.gather(*[
            agent.process(query) for agent in self.agents.values()
        ])
        
        # 协调员综合判断
        final_response = await self.agents["coordinator"].synthesize(
            query, expert_responses
        )
        
        return final_response
```

### Step 6: 记忆与上下文管理

#### 分层记忆架构
```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import FAISS
import redis

class MedicalMemorySystem:
    def __init__(self):
        # 短期记忆：当前对话
        self.short_term = ConversationSummaryBufferMemory(
            max_token_limit=1000,
            return_messages=True
        )
        
        # 中期记忆：会话摘要
        self.mid_term = redis.Redis(host='localhost', port=6379, db=0)
        
        # 长期记忆：知识向量库
        self.long_term = FAISS.from_documents(
            documents=[],  # 预加载医学知识库
            embedding=OpenAIEmbeddings()
        )
    
    async def store_interaction(self, query: MedicalQuery, response: MedicalResponse):
        # 短期记忆存储
        self.short_term.save_context(
            {"input": query.query_text},
            {"output": response.primary_response}
        )
        
        # 中期记忆：存储会话摘要
        session_id = self.get_session_id()
        summary = await self.generate_session_summary()
        self.mid_term.setex(f"session:{session_id}", 3600*24*7, summary)
        
        # 长期记忆：更新知识库
        if response.confidence_score > 0.8:
            await self.update_knowledge_base(query, response)
```

### Step 7: 多模态能力扩展

#### 多模态输入处理
```python
from langchain.document_loaders import ImageLoader
from langchain.schema import Document
import speech_recognition as sr
from PIL import Image

class MultimodalProcessor:
    def __init__(self):
        self.speech_recognizer = sr.Recognizer()
        self.image_analyzer = GPT4VisionAnalyzer()
    
    async def process_audio_query(self, audio_file: str) -> str:
        """语音转文字"""
        with sr.AudioFile(audio_file) as source:
            audio = self.speech_recognizer.record(source)
        return self.speech_recognizer.recognize_google(audio, language='zh-CN')
    
    async def analyze_medical_image(self, image_path: str, query: str) -> Dict:
        """医学影像分析"""
        image_analysis = await self.image_analyzer.analyze(
            image_path=image_path,
            prompt=f"作为医学影像专家，分析这张医学图像并回答：{query}"
        )
        
        return {
            "image_description": image_analysis.description,
            "medical_findings": image_analysis.findings,
            "recommendations": image_analysis.recommendations
        }
```

### Step 8: 输出格式与前端集成

#### Streamlit 前端界面
```python
import streamlit as st
import asyncio
from typing import Optional

class MedicalChatbot:
    def __init__(self):
        self.workflow = create_medical_workflow()
        self.memory = MedicalMemorySystem()
    
    def render_interface(self):
        st.set_page_config(
            page_title="MediAgents Pro",
            page_icon="🏥",
            layout="wide"
        )
        
        # 侧边栏配置
        with st.sidebar:
            st.header("配置选项")
            query_type = st.selectbox(
                "咨询类型", 
                ["diagnosis", "treatment", "research", "education"]
            )
            
            confidence_threshold = st.slider(
                "置信度阈值", 0.0, 1.0, 0.7, 0.1
            )
            
            include_sources = st.checkbox("包含参考来源", True)
        
        # 主界面
        st.title("🏥 MediAgents Pro - 智能医学助手")
        
        # 文件上传
        uploaded_files = st.file_uploader(
            "上传医学图像或报告",
            type=['jpg', 'png', 'pdf', 'txt'],
            accept_multiple_files=True
        )
        
        # 对话界面
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 用户输入
        if prompt := st.chat_input("请输入您的医学咨询..."):
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt
            })
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # AI 回复
            with st.chat_message("assistant"):
                response = await self.process_query(
                    prompt, query_type, uploaded_files
                )
                st.markdown(response.primary_response)
                
                # 显示置信度和来源
                if response.confidence_score < confidence_threshold:
                    st.warning(f"⚠️ 置信度较低 ({response.confidence_score:.2f})")
                
                if include_sources and response.sources:
                    with st.expander("参考来源"):
                        for source in response.sources:
                            st.write(f"- {source['title']}: {source['url']}")

# 运行应用
if __name__ == "__main__":
    chatbot = MedicalChatbot()
    chatbot.render_interface()
```

### Step 9: API 部署与监控

#### FastAPI 服务部署
```python
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="MediAgents API",
    description="高性能医学 AI 助手 API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v2/medical-query", response_model=MedicalResponse)
async def process_medical_query(
    query: MedicalQuery,
    files: Optional[List[UploadFile]] = File(None)
):
    try:
        # 处理上传的文件
        file_contexts = []
        if files:
            for file in files:
                file_context = await process_uploaded_file(file)
                file_contexts.append(file_context)
        
        # 处理查询
        workflow = create_medical_workflow()
        result = await workflow.ainvoke({
            "query": query,
            "context": MedicalContext(file_contexts=file_contexts)
        })
        
        return result["final_response"]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

# 性能监控
@app.middleware("http")
async def monitor_performance(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # 记录性能指标
    logger.info(f"Request processed in {process_time:.2f}s")
    
    return response
```

---

## 🛠️ 实施路线图

### 阶段 1: 基础架构迁移 (2-3 周)
- [ ] 安装 LangGraph 和相关依赖
- [ ] 重构现有代码为 StateGraph 工作流
- [ ] 实现基础的多 Agent 协作
- [ ] 添加输入输出验证

### 阶段 2: RAG 和记忆系统 (2-3 周)  
- [ ] 集成向量数据库和 RAG 检索
- [ ] 实现分层记忆架构
- [ ] 添加实时文献检索功能
- [ ] 优化知识库更新机制

### 阶段 3: 多模态和前端 (2-3 周)
- [ ] 添加图像和语音处理能力
- [ ] 开发 Streamlit 前端界面
- [ ] 实现文件上传和处理
- [ ] 优化用户体验和交互

### 阶段 4: API 和部署 (1-2 周)
- [ ] 开发 FastAPI 服务接口
- [ ] 添加性能监控和日志
- [ ] 实现负载均衡和缓存
- [ ] 部署到生产环境

### 阶段 5: 评估和优化 (持续)
- [ ] 实现 ROUGE 等评估指标
- [ ] 添加 A/B 测试框架
- [ ] 收集用户反馈和优化
- [ ] 持续模型微调和改进

---

## 📊 评估方法升级

### 多维度评估框架
```python
from rouge import Rouge
from bert_score import score
import numpy as np

class ComprehensiveEvaluator:
    def __init__(self):
        self.rouge = Rouge()
        
    def evaluate_response(self, 
                         reference: str, 
                         prediction: str,
                         query_type: str) -> Dict[str, float]:
        
        # ROUGE 评分
        rouge_scores = self.rouge.get_scores(prediction, reference)[0]
        
        # BERTScore 语义相似度
        P, R, F1 = score([prediction], [reference], lang="zh", verbose=False)
        
        # 领域特定评估
        domain_score = self.evaluate_domain_accuracy(
            prediction, reference, query_type
        )
        
        # 安全性评估
        safety_score = self.evaluate_safety(prediction)
        
        return {
            "rouge_1": rouge_scores['rouge-1']['f'],
            "rouge_2": rouge_scores['rouge-2']['f'], 
            "rouge_l": rouge_scores['rouge-l']['f'],
            "bert_score": F1.mean().item(),
            "domain_accuracy": domain_score,
            "safety_score": safety_score,
            "overall_score": self.calculate_weighted_score({
                "rouge": rouge_scores['rouge-l']['f'],
                "bert": F1.mean().item(),
                "domain": domain_score,
                "safety": safety_score
            })
        }
```

---

## ⚡ 性能优化策略

### 并行处理优化
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelProcessor:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def parallel_agent_processing(self, 
                                      query: MedicalQuery, 
                                      agents: List[Agent]) -> List[AgentResponse]:
        """并行处理多个 Agent"""
        tasks = [
            asyncio.create_task(agent.process(query)) 
            for agent in agents
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def batch_rag_retrieval(self, 
                                queries: List[str]) -> List[List[Document]]:
        """批量 RAG 检索"""
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(
                self.executor, 
                self.vectorstore.similarity_search, 
                query
            ) 
            for query in queries
        ]
        
        return await asyncio.gather(*tasks)
```

### 智能缓存策略
```python
from functools import lru_cache
import hashlib

class SmartCache:
    def __init__(self):
        self.redis_client = redis.Redis()
    
    def cache_key(self, query: str, context: dict) -> str:
        """生成缓存键"""
        content = f"{query}:{str(context)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def get_cached_response(self, cache_key: str) -> Optional[MedicalResponse]:
        """获取缓存响应"""
        cached = self.redis_client.get(f"response:{cache_key}")
        if cached:
            return MedicalResponse.parse_raw(cached)
        return None
    
    def cache_response(self, 
                      cache_key: str, 
                      response: MedicalResponse,
                      ttl: int = 3600):
        """缓存响应"""
        self.redis_client.setex(
            f"response:{cache_key}", 
            ttl, 
            response.json()
        )
```

---

这个优化指南基于九大步骤框架，提供了从架构重设计到部署上线的完整方案。重点关注 LangGraph 工作流、多模态能力、RAG 技术集成和生产就绪的部署方案。

建议按照阶段性路线图逐步实施，每个阶段都有明确的交付成果和测试标准。