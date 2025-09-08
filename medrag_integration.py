"""
MedRAG + LangGraph 集成方案
整合 MedRAG 工具包到 MediAgents-LangGraph 架构中
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from typing_extensions import Literal
from langgraph.graph import StateGraph, END
import operator
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import time
from pydantic import BaseModel, Field

# 导入现有组件
from langgraph_implementation import MedicalWorkflowState, MedicalQuery, QueryType, DifficultyLevel, AgentResponse
from utils import Agent, setup_model

# MedRAG 相关导入 (需要先安装 MedRAG)
# pip install git+https://github.com/Teddy-XiongGZ/MedRAG.git
try:
    # 假设 MedRAG 的主要组件
    from medrag import MedRAG
    from medrag.retrieval import BM25Retriever, ContrieverRetriever, MedCPTRetriever
    from medrag.corpora import PubMedCorpus, StatPearlsCorpus, WikipediaCorpus
    MEDRAG_AVAILABLE = True
except ImportError:
    print("⚠️ MedRAG 未安装，使用模拟实现")
    MEDRAG_AVAILABLE = False

@dataclass
class RAGResult:
    """RAG检索结果"""
    query: str
    retrieved_passages: List[Dict[str, Any]]
    relevance_scores: List[float]
    sources: List[Dict[str, str]]
    retrieval_time: float
    total_passages: int

class MedRAGConfig(BaseModel):
    """MedRAG 配置"""
    corpus_name: str = "pubmed"  # pubmed, statpearls, textbooks, wikipedia
    retriever_name: str = "medcpt"  # bm25, contriever, specter, medcpt
    retrieve_k: int = 5
    enable_rerank: bool = True
    max_passage_length: int = 500
    enable_followup_query: bool = True  # i-MedRAG feature

class EnhancedMedicalWorkflowState(MedicalWorkflowState):
    """增强的医学工作流状态 - 集成 RAG 功能"""
    # RAG 相关状态
    rag_results: Annotated[List[RAGResult], operator.add]
    knowledge_context: Optional[str]
    followup_queries: Annotated[List[str], operator.add]
    evidence_quality_score: Optional[float]

class MedRAGIntegratedSystem:
    """集成 MedRAG 的医学智能体系统"""
    
    def __init__(self, 
                 model_name: str = "gemini-2.5-flash-lite-preview-06-17",
                 medrag_config: Optional[MedRAGConfig] = None):
        self.model_name = model_name
        self.medrag_config = medrag_config or MedRAGConfig()
        
        # 设置模型
        self.setup_successful = setup_model(model_name)
        if not self.setup_successful:
            raise ValueError(f"Failed to setup model: {model_name}")
        
        # 初始化 MedRAG 组件
        self.medrag_system = self._initialize_medrag()
        
        # 创建增强的工作流
        self.workflow = self._create_enhanced_workflow()
    
    def _initialize_medrag(self) -> Optional[Any]:
        """初始化 MedRAG 系统"""
        if not MEDRAG_AVAILABLE:
            print("🔄 使用模拟 MedRAG 实现")
            return self._create_mock_medrag()
        
        try:
            print(f"🔄 初始化 MedRAG - Corpus: {self.medrag_config.corpus_name}, Retriever: {self.medrag_config.retriever_name}")
            
            # 选择语料库
            corpus_map = {
                "pubmed": PubMedCorpus,
                "statpearls": StatPearlsCorpus, 
                "wikipedia": WikipediaCorpus
            }
            
            # 选择检索器
            retriever_map = {
                "bm25": BM25Retriever,
                "contriever": ContrieverRetriever,
                "medcpt": MedCPTRetriever
            }
            
            corpus = corpus_map.get(self.medrag_config.corpus_name, PubMedCorpus)()
            retriever = retriever_map.get(self.medrag_config.retriever_name, MedCPTRetriever)()
            
            medrag_system = MedRAG(
                corpus=corpus,
                retriever=retriever,
                llm_name=self.model_name
            )
            
            print("✅ MedRAG 初始化成功")
            return medrag_system
            
        except Exception as e:
            print(f"⚠️ MedRAG 初始化失败: {e}")
            print("🔄 回退到模拟实现")
            return self._create_mock_medrag()
    
    def _create_mock_medrag(self):
        """创建 MedRAG 的模拟实现"""
        class MockMedRAG:
            def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[float]]:
                # 模拟检索结果
                mock_passages = [
                    f"Medical literature passage 1 relevant to: {query[:50]}...",
                    f"Clinical guideline excerpt 2 for: {query[:50]}...",
                    f"Research finding 3 related to: {query[:50]}...",
                    f"Treatment recommendation 4 about: {query[:50]}...",
                    f"Diagnostic criteria 5 for: {query[:50]}..."
                ]
                
                mock_scores = [0.95, 0.87, 0.82, 0.76, 0.71]
                return mock_passages[:k], mock_scores[:k]
            
            def generate_followup_queries(self, original_query: str) -> List[str]:
                # 模拟后续查询生成
                return [
                    f"What are the differential diagnoses for {original_query[:30]}?",
                    f"What are the latest treatment guidelines for {original_query[:30]}?",
                    f"What are the risk factors associated with {original_query[:30]}?"
                ]
        
        return MockMedRAG()
    
    def _create_enhanced_workflow(self) -> StateGraph:
        """创建增强的工作流，集成 RAG 功能"""
        workflow = StateGraph(EnhancedMedicalWorkflowState)
        
        # 添加节点 - 包含 RAG 增强节点
        workflow.add_node("difficulty_assessor", self._assess_difficulty)
        workflow.add_node("rag_retriever", self._retrieve_medical_knowledge)
        workflow.add_node("knowledge_synthesizer", self._synthesize_knowledge)
        workflow.add_node("expert_recruiter", self._recruit_rag_informed_experts)
        workflow.add_node("rag_enhanced_basic", self._rag_enhanced_basic_processing)
        workflow.add_node("rag_enhanced_intermediate", self._rag_enhanced_intermediate_processing)
        workflow.add_node("rag_enhanced_advanced", self._rag_enhanced_advanced_processing)
        workflow.add_node("evidence_evaluator", self._evaluate_evidence_quality)
        workflow.add_node("final_synthesizer", self._synthesize_final_response)
        workflow.add_node("quality_validator", self._validate_response_quality)
        
        # 设置入口点
        workflow.set_entry_point("difficulty_assessor")
        
        # 工作流路径：评估 -> RAG检索 -> 知识综合 -> 专家招募 -> 处理 -> 评估 -> 综合 -> 验证
        workflow.add_edge("difficulty_assessor", "rag_retriever")
        workflow.add_edge("rag_retriever", "knowledge_synthesizer") 
        workflow.add_edge("knowledge_synthesizer", "expert_recruiter")
        
        # 条件路由到不同的处理器
        workflow.add_conditional_edges(
            "expert_recruiter",
            self._route_to_rag_processor,
            {
                "basic": "rag_enhanced_basic",
                "intermediate": "rag_enhanced_intermediate",
                "advanced": "rag_enhanced_advanced"
            }
        )
        
        # 所有处理器都连接到证据评估
        workflow.add_edge("rag_enhanced_basic", "evidence_evaluator")
        workflow.add_edge("rag_enhanced_intermediate", "evidence_evaluator")
        workflow.add_edge("rag_enhanced_advanced", "evidence_evaluator")
        
        # 最终流程
        workflow.add_edge("evidence_evaluator", "final_synthesizer")
        workflow.add_edge("final_synthesizer", "quality_validator")
        workflow.add_edge("quality_validator", END)
        
        return workflow.compile()
    
    def _assess_difficulty(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """评估查询难度 - 复用原有逻辑"""
        print("[MedRAG-LangGraph] 🔍 评估查询难度...")
        
        query = state["query"]
        start_time = time.time()
        
        from utils import determine_difficulty
        difficulty_level, input_tokens, output_tokens = determine_difficulty(
            query.query_text, 
            "adaptive", 
            self.model_name
        )
        
        processing_time = time.time() - start_time
        
        print(f"[MedRAG-LangGraph] ✅ 难度评估: {difficulty_level} (耗时: {processing_time:.2f}s)")
        
        return {
            **state,
            "difficulty_level": DifficultyLevel(difficulty_level),
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "rag_results": [],
            "followup_queries": [],
            "processing_metadata": {
                "difficulty_assessment_time": processing_time
            }
        }
    
    def _retrieve_medical_knowledge(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """使用 MedRAG 检索医学知识"""
        print("[MedRAG-LangGraph] 🔍 检索医学知识...")
        
        query = state["query"]
        start_time = time.time()
        
        # 主查询检索
        passages, scores = self.medrag_system.retrieve(
            query.query_text, 
            k=self.medrag_config.retrieve_k
        )
        
        # 构建检索结果
        retrieved_passages = []
        sources = []
        
        for i, (passage, score) in enumerate(zip(passages, scores)):
            retrieved_passages.append({
                "content": passage,
                "score": score,
                "rank": i + 1,
                "source_type": self.medrag_config.corpus_name
            })
            
            sources.append({
                "title": f"{self.medrag_config.corpus_name.title()} Source {i+1}",
                "content": passage[:100] + "...",
                "score": score,
                "url": f"#{self.medrag_config.corpus_name}_source_{i+1}"
            })
        
        # 生成后续查询 (i-MedRAG 功能)
        followup_queries = []
        if self.medrag_config.enable_followup_query:
            followup_queries = self.medrag_system.generate_followup_queries(query.query_text)
        
        retrieval_time = time.time() - start_time
        
        rag_result = RAGResult(
            query=query.query_text,
            retrieved_passages=retrieved_passages,
            relevance_scores=scores,
            sources=sources,
            retrieval_time=retrieval_time,
            total_passages=len(passages)
        )
        
        print(f"[MedRAG-LangGraph] ✅ 知识检索完成: {len(passages)} 个相关片段 (耗时: {retrieval_time:.2f}s)")
        if followup_queries:
            print(f"[MedRAG-LangGraph] 💡 生成 {len(followup_queries)} 个后续查询")
        
        return {
            **state,
            "rag_results": [rag_result],
            "followup_queries": followup_queries
        }
    
    def _synthesize_knowledge(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """综合检索到的知识"""
        print("[MedRAG-LangGraph] 🧠 综合医学知识...")
        
        rag_results = state["rag_results"]
        if not rag_results:
            return {**state, "knowledge_context": ""}
        
        # 提取最相关的知识
        all_passages = []
        for result in rag_results:
            # 选择评分最高的片段
            top_passages = sorted(
                result.retrieved_passages, 
                key=lambda x: x["score"], 
                reverse=True
            )[:3]  # 取前3个最相关的
            
            all_passages.extend([p["content"] for p in top_passages])
        
        # 构建知识上下文
        knowledge_context = "\n\n".join([
            f"**Medical Knowledge {i+1}:**\n{passage}"
            for i, passage in enumerate(all_passages)
        ])
        
        print(f"[MedRAG-LangGraph] ✅ 知识综合完成: {len(all_passages)} 个关键知识片段")
        
        return {
            **state,
            "knowledge_context": knowledge_context
        }
    
    def _recruit_rag_informed_experts(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """基于 RAG 知识招募专家"""
        print(f"[MedRAG-LangGraph] 👥 基于检索知识招募 {state['difficulty_level']} 级别专家...")
        
        difficulty = state["difficulty_level"]
        query = state["query"]
        knowledge_context = state["knowledge_context"]
        
        # 基于检索到的知识优化专家配置
        expert_configs = self._get_rag_informed_expert_configs(difficulty, query, knowledge_context)
        
        recruited_experts = []
        for config in expert_configs:
            recruited_experts.append({
                "role": config["role"],
                "expertise": config["expertise"],
                "specialization": config.get("specialization", ""),
                "weight": config.get("weight", 1.0),
                "agent_id": f"{config['role']}_{len(recruited_experts)}",
                "rag_informed": True
            })
        
        print(f"[MedRAG-LangGraph] ✅ 招募完成: {len(recruited_experts)} 位知识增强专家")
        
        return {
            **state,
            "recruited_experts": recruited_experts
        }
    
    def _get_rag_informed_expert_configs(self, difficulty: DifficultyLevel, query: MedicalQuery, knowledge_context: str) -> List[Dict]:
        """基于RAG知识确定专家配置"""
        # 分析知识内容确定需要的专业领域
        knowledge_keywords = knowledge_context.lower()
        
        base_experts = [
            {"role": "RAG-Enhanced Clinician", "expertise": "evidence-based clinical decision making with access to latest medical literature"},
            {"role": "Knowledge Synthesizer", "expertise": "integrating multiple sources of medical evidence"},
            {"role": "Guidelines Specialist", "expertise": "interpreting and applying clinical practice guidelines"}
        ]
        
        # 根据知识内容添加专业化专家
        if any(term in knowledge_keywords for term in ["cardiac", "heart", "coronary"]):
            base_experts.append({"role": "Cardiologist", "expertise": "cardiovascular medicine and cardiac care"})
        
        if any(term in knowledge_keywords for term in ["neuro", "brain", "cognitive"]):
            base_experts.append({"role": "Neurologist", "expertise": "neurological disorders and brain health"})
        
        if any(term in knowledge_keywords for term in ["infection", "antibiotic", "pathogen"]):
            base_experts.append({"role": "Infectious Disease Specialist", "expertise": "infectious diseases and antimicrobial therapy"})
        
        # 根据难度调整专家数量
        if difficulty == DifficultyLevel.ADVANCED:
            base_experts.append({"role": "Research Physician", "expertise": "translating cutting-edge research into clinical practice"})
        
        return base_experts[:6]  # 限制专家数量
    
    def _rag_enhanced_basic_processing(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """RAG 增强的基础处理"""
        print("[MedRAG-LangGraph] 🔄 执行 RAG 增强基础处理...")
        
        query = state["query"]
        experts = state["recruited_experts"]
        knowledge_context = state["knowledge_context"]
        
        agent_responses = []
        total_input = total_output = 0
        
        for expert in experts:
            # 创建具有RAG知识的专家
            rag_enhanced_instruction = f"""
            You are a {expert['role']} with expertise in {expert['expertise']}.
            
            You have access to the following relevant medical knowledge:
            {knowledge_context}
            
            Use this evidence-based information to provide your professional medical opinion.
            """
            
            agent = Agent(
                instruction=rag_enhanced_instruction,
                role=expert['role'],
                model_info=self.model_name
            )
            
            start_time = time.time()
            response_text = agent.chat(
                f"Medical Query: {query.query_text}\n\nBased on the provided medical evidence, give your professional analysis and recommendation (limit to 200 words):"
            )
            processing_time = time.time() - start_time
            
            agent_response = AgentResponse(
                agent_id=expert['agent_id'],
                role=expert['role'],
                response=response_text,
                confidence=0.85,  # RAG增强后置信度提高
                tokens_used={
                    "input": agent.total_input_tokens,
                    "output": agent.total_output_tokens
                },
                processing_time=processing_time
            )
            
            agent_responses.append(agent_response)
            total_input += agent.total_input_tokens
            total_output += agent.total_output_tokens
        
        print(f"[MedRAG-LangGraph] ✅ RAG 增强基础处理完成: {len(agent_responses)} 个专家意见")
        
        return {
            **state,
            "agent_responses": agent_responses,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output
        }
    
    def _rag_enhanced_intermediate_processing(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """RAG 增强的中级处理"""
        print("[MedRAG-LangGraph] 🔄 执行 RAG 增强中级处理...")
        
        # 结合 RAG 知识的专家协作讨论
        query = state["query"]
        experts = state["recruited_experts"] 
        knowledge_context = state["knowledge_context"]
        
        # 第一轮：基于RAG知识的初始意见
        initial_responses = self._rag_informed_initial_opinions(query, experts, knowledge_context)
        
        # 第二轮：知识增强的专家讨论
        discussion_responses = self._rag_enhanced_expert_discussion(query, experts, initial_responses, knowledge_context)
        
        all_responses = initial_responses + discussion_responses
        
        total_input = sum(r.tokens_used["input"] for r in all_responses)
        total_output = sum(r.tokens_used["output"] for r in all_responses)
        
        print(f"[MedRAG-LangGraph] ✅ RAG 增强中级处理完成: {len(all_responses)} 轮专家交互")
        
        return {
            **state,
            "agent_responses": all_responses,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "interaction_history": [
                {"round": 1, "type": "rag_informed_opinions", "count": len(initial_responses)},
                {"round": 2, "type": "rag_enhanced_discussion", "count": len(discussion_responses)}
            ]
        }
    
    def _rag_enhanced_advanced_processing(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """RAG 增强的高级处理"""
        print("[MedRAG-LangGraph] 🔄 执行 RAG 增强高级处理...")
        
        query = state["query"]
        experts = state["recruited_experts"]
        knowledge_context = state["knowledge_context"]
        
        # 多阶段处理，每个阶段都使用 RAG 知识
        responses = []
        
        # 阶段1：专业分析 (基于RAG)
        specialist_responses = self._rag_specialist_analysis(query, experts, knowledge_context)
        responses.extend(specialist_responses)
        
        # 阶段2：跨学科讨论 (RAG增强)
        interdisciplinary_responses = self._rag_interdisciplinary_discussion(
            query, experts, specialist_responses, knowledge_context
        )
        responses.extend(interdisciplinary_responses)
        
        # 阶段3：协调员综合 (整合所有RAG知识)
        coordinator_response = self._rag_coordinator_synthesis(
            query, experts, responses, knowledge_context
        )
        responses.append(coordinator_response)
        
        total_input = sum(r.tokens_used["input"] for r in responses)
        total_output = sum(r.tokens_used["output"] for r in responses)
        
        print(f"[MedRAG-LangGraph] ✅ RAG 增强高级处理完成: 3 阶段共 {len(responses)} 次交互")
        
        return {
            **state,
            "agent_responses": responses,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "interaction_history": [
                {"phase": 1, "type": "rag_specialist_analysis", "count": len(specialist_responses)},
                {"phase": 2, "type": "rag_interdisciplinary_discussion", "count": len(interdisciplinary_responses)},
                {"phase": 3, "type": "rag_coordinator_synthesis", "count": 1}
            ]
        }
    
    def _evaluate_evidence_quality(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """评估证据质量"""
        print("[MedRAG-LangGraph] 📊 评估证据质量...")
        
        rag_results = state["rag_results"]
        agent_responses = state["agent_responses"]
        
        # 计算证据质量评分
        evidence_scores = []
        
        if rag_results:
            for result in rag_results:
                # 基于检索评分和数量计算质量
                avg_relevance = sum(result.relevance_scores) / len(result.relevance_scores)
                passage_count_factor = min(result.total_passages / 5, 1.0)  # 归一化到[0,1]
                
                quality_score = (avg_relevance * 0.7) + (passage_count_factor * 0.3)
                evidence_scores.append(quality_score)
        
        # 综合证据质量评分
        overall_evidence_quality = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.5
        
        print(f"[MedRAG-LangGraph] 📊 证据质量评估完成: {overall_evidence_quality:.2f}")
        
        return {
            **state,
            "evidence_quality_score": overall_evidence_quality
        }
    
    def _synthesize_final_response(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """综合最终响应 - RAG增强版"""
        print("[MedRAG-LangGraph] 🎯 综合 RAG 增强的最终医学建议...")
        
        query = state["query"]
        agent_responses = state["agent_responses"] 
        knowledge_context = state["knowledge_context"]
        evidence_quality = state["evidence_quality_score"]
        rag_results = state["rag_results"]
        
        # 创建RAG增强的综合智能体
        rag_enhanced_synthesizer = Agent(
            instruction=f"""
            You are a senior medical consultant with access to comprehensive medical literature and evidence.
            
            Available Medical Evidence:
            {knowledge_context}
            
            Your role is to synthesize multiple expert opinions with evidence-based medical knowledge to provide the most accurate and comprehensive medical recommendation.
            """,
            role="RAG-Enhanced Medical Synthesizer",
            model_info=self.model_name
        )
        
        # 构建综合提示
        expert_opinions = "\n\n".join([
            f"**{resp.role} Opinion (Confidence: {resp.confidence:.2f}):**\n{resp.response}"
            for resp in agent_responses
        ])
        
        sources_summary = ""
        if rag_results and rag_results[0].sources:
            sources_summary = "\n\n**Evidence Sources:**\n" + "\n".join([
                f"- {source['title']}: {source['content']}"
                for source in rag_results[0].sources[:3]  # 显示前3个来源
            ])
        
        synthesis_prompt = f"""
        Medical Query: {query.query_text}
        
        Expert Opinions:
        {expert_opinions}
        
        {sources_summary}
        
        Evidence Quality Score: {evidence_quality:.2f}/1.0
        
        Based on the expert opinions and the provided medical evidence, provide a comprehensive, evidence-based medical recommendation that:
        
        1. **Clinical Assessment**: Synthesize key diagnostic and clinical insights
        2. **Evidence-Based Recommendations**: Provide recommendations backed by the available evidence
        3. **Risk Considerations**: Address any potential risks or contraindications
        4. **Follow-up Actions**: Suggest appropriate follow-up measures
        5. **Limitations**: Acknowledge any limitations in the current evidence
        
        Structure your response clearly and limit to 500 words. Include confidence indicators where appropriate.
        """
        
        start_time = time.time()
        final_response = rag_enhanced_synthesizer.chat(synthesis_prompt)
        synthesis_time = time.time() - start_time
        
        # RAG增强的置信度计算
        base_confidence = sum(r.confidence for r in agent_responses) / len(agent_responses)
        evidence_boost = evidence_quality * 0.1  # 证据质量带来的置信度提升
        final_confidence = min(base_confidence + evidence_boost, 0.95)
        
        print(f"[MedRAG-LangGraph] ✅ RAG 增强最终建议完成 (置信度: {final_confidence:.2f})")
        
        return {
            **state,
            "final_decision": final_response,
            "confidence_score": final_confidence,
            "total_input_tokens": rag_enhanced_synthesizer.total_input_tokens,
            "total_output_tokens": rag_enhanced_synthesizer.total_output_tokens,
            "processing_metadata": {
                **state.get("processing_metadata", {}),
                "synthesis_time": synthesis_time,
                "evidence_enhanced": True,
                "evidence_quality_score": evidence_quality,
                "total_sources": len(rag_results[0].sources) if rag_results else 0
            }
        }
    
    def _validate_response_quality(self, state: EnhancedMedicalWorkflowState) -> EnhancedMedicalWorkflowState:
        """RAG增强的质量验证"""
        print("[MedRAG-LangGraph] ✅ 执行 RAG 增强质量验证...")
        
        final_decision = state["final_decision"]
        confidence = state["confidence_score"] 
        evidence_quality = state["evidence_quality_score"]
        
        # 增强的质量指标
        quality_metrics = {
            "response_length": len(final_decision.split()),
            "confidence_score": confidence,
            "evidence_quality_score": evidence_quality,
            "expert_count": len(state["agent_responses"]),
            "sources_count": len(state["rag_results"][0].sources) if state["rag_results"] else 0,
            "processing_complete": final_decision is not None,
            "rag_enhanced": True
        }
        
        # RAG增强的质量检查标准
        quality_passed = (
            quality_metrics["response_length"] > 100 and
            quality_metrics["confidence_score"] > 0.6 and
            quality_metrics["evidence_quality_score"] > 0.3 and
            quality_metrics["sources_count"] > 0 and
            quality_metrics["processing_complete"]
        )
        
        print(f"[MedRAG-LangGraph] 📊 RAG 增强质量验证 - {'通过' if quality_passed else '需要改进'}")
        
        return {
            **state,
            "processing_metadata": {
                **state.get("processing_metadata", {}),
                "quality_metrics": quality_metrics,
                "quality_passed": quality_passed,
                "rag_enhanced_validation": True
            }
        }
    
    # 路由函数
    def _route_to_rag_processor(self, state: EnhancedMedicalWorkflowState) -> Literal["basic", "intermediate", "advanced"]:
        """路由到RAG增强的处理器"""
        difficulty = state["difficulty_level"]
        print(f"[MedRAG-LangGraph] 🔀 路由到 RAG 增强 {difficulty.value} 处理器")
        return difficulty.value
    
    # RAG增强的辅助方法
    def _rag_informed_initial_opinions(self, query: MedicalQuery, experts: List[Dict], knowledge_context: str) -> List[AgentResponse]:
        """基于RAG知识的初始意见收集"""
        responses = []
        for expert in experts:
            agent = Agent(
                instruction=f"""
                You are a {expert['role']} with expertise in {expert['expertise']}.
                
                Relevant Medical Evidence:
                {knowledge_context}
                
                Base your analysis on the provided evidence and your professional expertise.
                """,
                role=expert['role'],
                model_info=self.model_name
            )
            
            start_time = time.time()
            response_text = agent.chat(
                f"Medical Query: {query.query_text}\n\nProvide your evidence-based analysis (limit to 150 words):"
            )
            processing_time = time.time() - start_time
            
            responses.append(AgentResponse(
                agent_id=expert['agent_id'],
                role=expert['role'],
                response=response_text,
                confidence=0.8,  # RAG增强提高置信度
                tokens_used={"input": agent.total_input_tokens, "output": agent.total_output_tokens},
                processing_time=processing_time
            ))
        
        return responses
    
    def _rag_enhanced_expert_discussion(self, query: MedicalQuery, experts: List[Dict], 
                                       initial_responses: List[AgentResponse], 
                                       knowledge_context: str) -> List[AgentResponse]:
        """RAG增强的专家讨论"""
        discussion_responses = []
        
        other_opinions = "\n".join([
            f"{resp.role}: {resp.response[:100]}..."
            for resp in initial_responses
        ])
        
        for i, expert in enumerate(experts):
            agent = Agent(
                instruction=f"""
                You are a {expert['role']} in a medical consultation with access to comprehensive medical literature.
                
                Available Evidence:
                {knowledge_context}
                """,
                role=expert['role'],
                model_info=self.model_name
            )
            
            discussion_prompt = f"""
            Medical Query: {query.query_text}
            
            Colleague Opinions:
            {other_opinions}
            
            Based on the evidence and your colleagues' insights, provide your refined analysis (limit to 100 words):
            """
            
            start_time = time.time()
            response_text = agent.chat(discussion_prompt)
            processing_time = time.time() - start_time
            
            discussion_responses.append(AgentResponse(
                agent_id=f"{expert['agent_id']}_discussion",
                role=f"{expert['role']} (Evidence-Enhanced Discussion)",
                response=response_text,
                confidence=0.85,
                tokens_used={"input": agent.total_input_tokens, "output": agent.total_output_tokens},
                processing_time=processing_time
            ))
        
        return discussion_responses
    
    # 其他 RAG 增强方法的实现...
    def _rag_specialist_analysis(self, query, experts, knowledge_context):
        return self._rag_informed_initial_opinions(query, experts, knowledge_context)
    
    def _rag_interdisciplinary_discussion(self, query, experts, previous_responses, knowledge_context):
        return self._rag_enhanced_expert_discussion(query, experts, previous_responses, knowledge_context)
    
    def _rag_coordinator_synthesis(self, query, experts, all_responses, knowledge_context):
        coordinator = Agent(
            instruction=f"""
            You are a senior medical coordinator with access to comprehensive medical evidence.
            
            Available Evidence:
            {knowledge_context}
            """,
            role="RAG-Enhanced MDT Coordinator",
            model_info=self.model_name
        )
        
        all_opinions = "\n\n".join([
            f"**{resp.role}:**\n{resp.response}"
            for resp in all_responses
        ])
        
        coordinator_prompt = f"""
        Medical Query: {query.query_text}
        
        Team Analysis:
        {all_opinions}
        
        As coordinator with access to medical evidence, provide final evidence-based recommendation (limit to 200 words):
        """
        
        start_time = time.time()
        response_text = coordinator.chat(coordinator_prompt)
        processing_time = time.time() - start_time
        
        return AgentResponse(
            agent_id="rag_enhanced_coordinator",
            role="RAG-Enhanced MDT Coordinator",
            response=response_text,
            confidence=0.9,
            tokens_used={"input": coordinator.total_input_tokens, "output": coordinator.total_output_tokens},
            processing_time=processing_time
        )
    
    async def process_medical_query(self, query: MedicalQuery) -> Dict[str, Any]:
        """处理医学查询的主入口 - RAG增强版"""
        print(f"\n[MedRAG-LangGraph] 🚀 开始 RAG 增强医学查询处理: {query.query_text[:50]}...")
        
        # 初始化增强状态
        initial_state = EnhancedMedicalWorkflowState(
            query=query,
            difficulty_level=None,
            recruited_experts=[],
            agent_responses=[],
            interaction_history=[],
            final_decision=None,
            confidence_score=None,
            processing_metadata=None,
            total_input_tokens=0,
            total_output_tokens=0,
            rag_results=[],
            knowledge_context=None,
            followup_queries=[],
            evidence_quality_score=None
        )
        
        # 运行RAG增强工作流
        start_time = time.time()
        final_state = self.workflow.invoke(initial_state)
        total_time = time.time() - start_time
        
        # 构建增强的返回结果
        result = {
            "query_id": f"medrag_{int(time.time())}",
            "query": query.query_text,
            "difficulty_level": final_state["difficulty_level"].value,
            "final_decision": final_state["final_decision"],
            "confidence_score": final_state["confidence_score"],
            "evidence_quality_score": final_state["evidence_quality_score"],
            "expert_count": len(final_state["agent_responses"]),
            "sources_count": len(final_state["rag_results"][0].sources) if final_state["rag_results"] else 0,
            "followup_queries": final_state["followup_queries"],
            "token_usage": {
                "input_tokens": final_state["total_input_tokens"],
                "output_tokens": final_state["total_output_tokens"],
                "total_tokens": final_state["total_input_tokens"] + final_state["total_output_tokens"]
            },
            "processing_time": total_time,
            "rag_enhanced": True,
            "retrieval_info": {
                "retrieval_time": final_state["rag_results"][0].retrieval_time if final_state["rag_results"] else 0,
                "passages_retrieved": final_state["rag_results"][0].total_passages if final_state["rag_results"] else 0,
                "corpus_used": self.medrag_config.corpus_name,
                "retriever_used": self.medrag_config.retriever_name
            },
            "sources": final_state["rag_results"][0].sources if final_state["rag_results"] else [],
            "metadata": final_state["processing_metadata"]
        }
        
        print(f"\n[MedRAG-LangGraph] ✨ RAG 增强处理完成! 耗时: {total_time:.2f}s")
        print(f"[MedRAG-LangGraph] 📚 检索到 {result['sources_count']} 个医学知识源")
        print(f"[MedRAG-LangGraph] 📊 证据质量: {result['evidence_quality_score']:.2f}")
        print(f"[MedRAG-LangGraph] 🎯 最终置信度: {result['confidence_score']:.2f}")
        
        return result

# 使用示例
if __name__ == "__main__":
    # 配置 MedRAG
    medrag_config = MedRAGConfig(
        corpus_name="pubmed",
        retriever_name="medcpt", 
        retrieve_k=5,
        enable_rerank=True,
        enable_followup_query=True
    )
    
    # 初始化系统
    rag_system = MedRAGIntegratedSystem(
        model_name="gemini-2.5-flash-lite-preview-06-17",
        medrag_config=medrag_config
    )
    
    # 测试查询
    test_query = MedicalQuery(
        query_text="A 65-year-old patient with diabetes presents with acute chest pain. Blood pressure is 180/100. What is the optimal emergency management approach?",
        query_type=QueryType.DIAGNOSIS
    )
    
    # 处理查询
    result = asyncio.run(rag_system.process_medical_query(test_query))
    
    # 输出结果
    print("\n" + "="*80)
    print("🏥 MEDRAG-LANGGRAPH 处理结果")
    print("="*80)
    print(f"查询ID: {result['query_id']}")
    print(f"难度级别: {result['difficulty_level']}")
    print(f"专家数量: {result['expert_count']}")
    print(f"知识源数量: {result['sources_count']}")
    print(f"证据质量: {result['evidence_quality_score']:.2f}")
    print(f"处理时间: {result['processing_time']:.2f}秒")
    print(f"检索耗时: {result['retrieval_info']['retrieval_time']:.2f}秒")
    print(f"最终置信度: {result['confidence_score']:.2f}")
    print(f"Token使用: {result['token_usage']['total_tokens']}")
    
    print(f"\n📚 使用的知识库: {result['retrieval_info']['corpus_used']}")
    print(f"🔍 检索器: {result['retrieval_info']['retriever_used']}")
    print(f"📄 检索片段数: {result['retrieval_info']['passages_retrieved']}")
    
    if result['followup_queries']:
        print(f"\n💡 建议的后续查询:")
        for i, query in enumerate(result['followup_queries'], 1):
            print(f"   {i}. {query}")
    
    print(f"\n🎯 最终 RAG 增强建议:")
    print("-" * 60)
    print(result['final_decision'])
    
    if result['sources']:
        print(f"\n📖 参考来源:")
        for i, source in enumerate(result['sources'][:3], 1):  # 显示前3个来源
            print(f"   {i}. {source['title']} (评分: {source.get('score', 'N/A')})")
    
    print("="*80)