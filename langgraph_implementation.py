"""
MediAgents-LangGraph 实现方案
基于现有架构的 LangGraph 重构示例
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from typing_extensions import Literal
from langgraph.graph import StateGraph, END
import operator
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import time
from pydantic import BaseModel, Field

# 导入现有的工具函数
from utils import Agent, setup_model, determine_difficulty

class DifficultyLevel(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"

class QueryType(str, Enum):
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    RESEARCH = "research"
    EDUCATION = "education"

@dataclass
class AgentResponse:
    agent_id: str
    role: str
    response: str
    confidence: float
    tokens_used: Dict[str, int]
    processing_time: float

@dataclass
class MedicalQuery:
    query_text: str
    query_type: QueryType = QueryType.DIAGNOSIS
    context: Optional[Dict[str, Any]] = None
    priority: int = 1

class MedicalWorkflowState(TypedDict):
    """LangGraph 工作流状态定义"""
    # 输入状态
    query: MedicalQuery
    difficulty_level: Optional[DifficultyLevel]
    
    # 处理过程状态
    recruited_experts: Annotated[List[Dict], operator.add]
    agent_responses: Annotated[List[AgentResponse], operator.add]
    interaction_history: Annotated[List[Dict], operator.add]
    
    # 输出状态
    final_decision: Optional[str]
    confidence_score: Optional[float]
    processing_metadata: Optional[Dict[str, Any]]
    
    # Token 使用跟踪
    total_input_tokens: Annotated[int, operator.add]
    total_output_tokens: Annotated[int, operator.add]

class MediAgentsLangGraph:
    """基于 LangGraph 的 MediAgents 系统"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        self.model_name = model_name
        self.setup_successful = setup_model(model_name)
        
        if not self.setup_successful:
            raise ValueError(f"Failed to setup model: {model_name}")
        
        # 创建工作流图
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """创建 LangGraph 工作流"""
        workflow = StateGraph(MedicalWorkflowState)
        
        # 添加节点
        workflow.add_node("difficulty_assessor", self._assess_difficulty)
        workflow.add_node("expert_recruiter", self._recruit_experts)
        workflow.add_node("basic_processor", self._process_basic_workflow)
        workflow.add_node("intermediate_processor", self._process_intermediate_workflow)
        workflow.add_node("advanced_processor", self._process_advanced_workflow)
        workflow.add_node("response_synthesizer", self._synthesize_response)
        workflow.add_node("quality_validator", self._validate_quality)
        
        # 设置入口点
        workflow.set_entry_point("difficulty_assessor")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "difficulty_assessor",
            self._route_by_difficulty,
            {
                "basic": "expert_recruiter",
                "intermediate": "expert_recruiter", 
                "advanced": "expert_recruiter"
            }
        )
        
        workflow.add_conditional_edges(
            "expert_recruiter",
            self._route_to_processor,
            {
                "basic": "basic_processor",
                "intermediate": "intermediate_processor",
                "advanced": "advanced_processor"
            }
        )
        
        # 添加普通边
        workflow.add_edge("basic_processor", "response_synthesizer")
        workflow.add_edge("intermediate_processor", "response_synthesizer") 
        workflow.add_edge("advanced_processor", "response_synthesizer")
        workflow.add_edge("response_synthesizer", "quality_validator")
        workflow.add_edge("quality_validator", END)
        
        return workflow.compile()
    
    def _assess_difficulty(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """评估查询难度"""
        print("[LangGraph] 🔍 评估查询难度...")
        
        query = state["query"]
        start_time = time.time()
        
        # 使用现有的难度评估函数
        difficulty_level, input_tokens, output_tokens = determine_difficulty(
            query.query_text, 
            "adaptive", 
            self.model_name
        )
        
        processing_time = time.time() - start_time
        
        print(f"[LangGraph] ✅ 难度评估完成: {difficulty_level} (耗时: {processing_time:.2f}s)")
        
        return {
            **state,
            "difficulty_level": DifficultyLevel(difficulty_level),
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "processing_metadata": {
                "difficulty_assessment_time": processing_time
            }
        }
    
    def _recruit_experts(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """招募专家团队"""
        print(f"[LangGraph] 👥 招募 {state['difficulty_level']} 级别专家团队...")
        
        difficulty = state["difficulty_level"]
        query = state["query"]
        
        # 根据难度级别确定专家配置
        expert_configs = self._get_expert_configs(difficulty, query)
        
        recruited_experts = []
        for config in expert_configs:
            recruited_experts.append({
                "role": config["role"],
                "expertise": config["expertise"],
                "weight": config.get("weight", 1.0),
                "agent_id": f"{config['role']}_{len(recruited_experts)}"
            })
        
        print(f"[LangGraph] ✅ 成功招募 {len(recruited_experts)} 位专家")
        
        return {
            **state,
            "recruited_experts": recruited_experts
        }
    
    def _get_expert_configs(self, difficulty: DifficultyLevel, query: MedicalQuery) -> List[Dict]:
        """根据难度和查询类型获取专家配置"""
        base_configs = {
            DifficultyLevel.BASIC: [
                {"role": "General Practitioner", "expertise": "comprehensive medical knowledge and clinical experience"},
                {"role": "Specialist", "expertise": "specialized medical knowledge in relevant field"},
                {"role": "Clinical Researcher", "expertise": "evidence-based medicine and research methodology"}
            ],
            DifficultyLevel.INTERMEDIATE: [
                {"role": "Senior Clinician", "expertise": "advanced clinical decision-making"},
                {"role": "Medical Specialist", "expertise": "domain-specific expert knowledge"},
                {"role": "Research Physician", "expertise": "latest medical research and guidelines"},
                {"role": "Consultant", "expertise": "multi-disciplinary medical consultation"}
            ],
            DifficultyLevel.ADVANCED: [
                {"role": "Department Head", "expertise": "comprehensive medical leadership and expertise"},
                {"role": "Research Director", "expertise": "cutting-edge medical research and innovation"},
                {"role": "Multi-disciplinary Coordinator", "expertise": "coordinating complex medical teams"},
                {"role": "Ethics Consultant", "expertise": "medical ethics and complex decision-making"}
            ]
        }
        
        return base_configs.get(difficulty, base_configs[DifficultyLevel.BASIC])
    
    def _process_basic_workflow(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """处理基础级别查询 - 简化的专家仲裁模式"""
        print("[LangGraph] 🔄 执行基础级别处理流程...")
        
        query = state["query"]
        experts = state["recruited_experts"]
        
        agent_responses = []
        total_input = total_output = 0
        
        # 创建专家智能体并获取独立意见
        for expert in experts:
            agent = Agent(
                instruction=f"You are a {expert['role']} with expertise in {expert['expertise']}. Provide your professional medical opinion.",
                role=expert['role'],
                model_info=self.model_name
            )
            
            start_time = time.time()
            response_text = agent.chat(
                f"Medical Query: {query.query_text}\n\nProvide your professional analysis and recommendation (limit to 200 words):"
            )
            processing_time = time.time() - start_time
            
            agent_response = AgentResponse(
                agent_id=expert['agent_id'],
                role=expert['role'], 
                response=response_text,
                confidence=0.8,  # 可以后续改进为动态计算
                tokens_used={
                    "input": agent.total_input_tokens,
                    "output": agent.total_output_tokens
                },
                processing_time=processing_time
            )
            
            agent_responses.append(agent_response)
            total_input += agent.total_input_tokens
            total_output += agent.total_output_tokens
        
        print(f"[LangGraph] ✅ 基础处理完成，收集 {len(agent_responses)} 个专家意见")
        
        return {
            **state,
            "agent_responses": agent_responses,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output
        }
    
    def _process_intermediate_workflow(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """处理中级查询 - 专家协作讨论模式"""
        print("[LangGraph] 🔄 执行中级处理流程...")
        
        query = state["query"]
        experts = state["recruited_experts"]
        
        # 第一轮：收集初始意见
        initial_responses = self._collect_initial_opinions(query, experts)
        
        # 第二轮：基于其他专家意见进行讨论
        discussion_responses = self._conduct_expert_discussion(query, experts, initial_responses)
        
        # 合并所有响应
        all_responses = initial_responses + discussion_responses
        
        # 计算总token使用量
        total_input = sum(r.tokens_used["input"] for r in all_responses)
        total_output = sum(r.tokens_used["output"] for r in all_responses)
        
        print(f"[LangGraph] ✅ 中级处理完成，进行了 {len(all_responses)} 轮专家交互")
        
        return {
            **state,
            "agent_responses": all_responses,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "interaction_history": [
                {"round": 1, "type": "initial_opinions", "count": len(initial_responses)},
                {"round": 2, "type": "discussion", "count": len(discussion_responses)}
            ]
        }
    
    def _process_advanced_workflow(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """处理高级查询 - 多学科团队协调模式"""
        print("[LangGraph] 🔄 执行高级处理流程...")
        
        query = state["query"]
        experts = state["recruited_experts"]
        
        # 高级流程：多轮讨论 + 协调员综合
        responses = []
        
        # 第一阶段：专业分析
        specialist_responses = self._specialist_analysis_phase(query, experts)
        responses.extend(specialist_responses)
        
        # 第二阶段：跨学科讨论
        interdisciplinary_responses = self._interdisciplinary_discussion_phase(
            query, experts, specialist_responses
        )
        responses.extend(interdisciplinary_responses)
        
        # 第三阶段：协调员综合
        coordinator_response = self._coordinator_synthesis_phase(
            query, experts, responses
        )
        responses.append(coordinator_response)
        
        # 计算总token使用量
        total_input = sum(r.tokens_used["input"] for r in responses)
        total_output = sum(r.tokens_used["output"] for r in responses)
        
        print(f"[LangGraph] ✅ 高级处理完成，完成 3 阶段共 {len(responses)} 次专家交互")
        
        return {
            **state,
            "agent_responses": responses,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "interaction_history": [
                {"phase": 1, "type": "specialist_analysis", "count": len(specialist_responses)},
                {"phase": 2, "type": "interdisciplinary_discussion", "count": len(interdisciplinary_responses)},
                {"phase": 3, "type": "coordinator_synthesis", "count": 1}
            ]
        }
    
    def _synthesize_response(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """综合最终响应"""
        print("[LangGraph] 🎯 综合最终医学建议...")
        
        query = state["query"]
        agent_responses = state["agent_responses"]
        difficulty = state["difficulty_level"]
        
        # 创建综合智能体
        synthesizer = Agent(
            instruction="You are a senior medical consultant responsible for synthesizing multiple expert opinions into a comprehensive, actionable medical recommendation.",
            role="Medical Synthesizer",
            model_info=self.model_name
        )
        
        # 构建综合提示
        expert_opinions = "\n\n".join([
            f"**{resp.role} Opinion:**\n{resp.response}"
            for resp in agent_responses
        ])
        
        synthesis_prompt = f"""
        Medical Query: {query.query_text}
        
        Expert Opinions to Synthesize:
        {expert_opinions}
        
        Please provide a comprehensive, well-structured medical recommendation that:
        1. Synthesizes the key insights from all expert opinions
        2. Addresses any conflicting viewpoints with evidence-based reasoning
        3. Provides clear, actionable recommendations
        4. Includes appropriate caveats and follow-up suggestions
        
        Structure your response with clear sections and limit to 400 words.
        """
        
        start_time = time.time()
        final_response = synthesizer.chat(synthesis_prompt)
        synthesis_time = time.time() - start_time
        
        # 计算综合置信度 (简化版本)
        avg_confidence = sum(r.confidence for r in agent_responses) / len(agent_responses)
        confidence_score = min(avg_confidence * 0.95, 0.95)  # 轻微降低以反映综合的不确定性
        
        print(f"[LangGraph] ✅ 最终建议综合完成 (置信度: {confidence_score:.2f})")
        
        return {
            **state,
            "final_decision": final_response,
            "confidence_score": confidence_score,
            "total_input_tokens": synthesizer.total_input_tokens,
            "total_output_tokens": synthesizer.total_output_tokens,
            "processing_metadata": {
                **state.get("processing_metadata", {}),
                "synthesis_time": synthesis_time,
                "total_expert_responses": len(agent_responses)
            }
        }
    
    def _validate_quality(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """质量验证和最终检查"""
        print("[LangGraph] ✅ 执行质量验证...")
        
        final_decision = state["final_decision"]
        confidence = state["confidence_score"]
        
        # 简单的质量检查
        quality_metrics = {
            "response_length": len(final_decision.split()),
            "confidence_score": confidence,
            "expert_count": len(state["agent_responses"]),
            "processing_complete": final_decision is not None
        }
        
        # 可以添加更复杂的质量验证逻辑
        quality_passed = (
            quality_metrics["response_length"] > 50 and
            quality_metrics["confidence_score"] > 0.5 and
            quality_metrics["processing_complete"]
        )
        
        print(f"[LangGraph] 📊 质量验证完成 - {'通过' if quality_passed else '需要改进'}")
        
        return {
            **state,
            "processing_metadata": {
                **state.get("processing_metadata", {}),
                "quality_metrics": quality_metrics,
                "quality_passed": quality_passed
            }
        }
    
    # 路由函数
    def _route_by_difficulty(self, state: MedicalWorkflowState) -> Literal["basic", "intermediate", "advanced"]:
        """根据难度路由到不同处理流程"""
        difficulty = state["difficulty_level"]
        print(f"[LangGraph] 🔀 路由到 {difficulty.value} 处理流程")
        return difficulty.value
    
    def _route_to_processor(self, state: MedicalWorkflowState) -> Literal["basic", "intermediate", "advanced"]:
        """路由到具体的处理器"""
        return state["difficulty_level"].value
    
    # 辅助方法
    def _collect_initial_opinions(self, query: MedicalQuery, experts: List[Dict]) -> List[AgentResponse]:
        """收集专家初始意见"""
        responses = []
        for expert in experts:
            agent = Agent(
                instruction=f"You are a {expert['role']} with expertise in {expert['expertise']}.",
                role=expert['role'],
                model_info=self.model_name
            )
            
            start_time = time.time()
            response_text = agent.chat(
                f"Medical Query: {query.query_text}\n\nProvide your initial professional analysis (limit to 150 words):"
            )
            processing_time = time.time() - start_time
            
            responses.append(AgentResponse(
                agent_id=expert['agent_id'],
                role=expert['role'],
                response=response_text,
                confidence=0.75,
                tokens_used={"input": agent.total_input_tokens, "output": agent.total_output_tokens},
                processing_time=processing_time
            ))
        
        return responses
    
    def _conduct_expert_discussion(self, query: MedicalQuery, experts: List[Dict], initial_responses: List[AgentResponse]) -> List[AgentResponse]:
        """专家讨论阶段"""
        discussion_responses = []
        
        # 构建其他专家意见的摘要
        other_opinions = "\n".join([
            f"{resp.role}: {resp.response[:100]}..."
            for resp in initial_responses
        ])
        
        for i, expert in enumerate(experts):
            agent = Agent(
                instruction=f"You are a {expert['role']} participating in a medical consultation discussion.",
                role=expert['role'],
                model_info=self.model_name
            )
            
            discussion_prompt = f"""
            Medical Query: {query.query_text}
            
            Other Expert Opinions:
            {other_opinions}
            
            Based on the other experts' initial opinions, provide your refined analysis and any additional insights (limit to 100 words):
            """
            
            start_time = time.time()
            response_text = agent.chat(discussion_prompt)
            processing_time = time.time() - start_time
            
            discussion_responses.append(AgentResponse(
                agent_id=f"{expert['agent_id']}_discussion",
                role=f"{expert['role']} (Discussion)",
                response=response_text,
                confidence=0.8,
                tokens_used={"input": agent.total_input_tokens, "output": agent.total_output_tokens},
                processing_time=processing_time
            ))
        
        return discussion_responses
    
    def _specialist_analysis_phase(self, query: MedicalQuery, experts: List[Dict]) -> List[AgentResponse]:
        """专家分析阶段"""
        return self._collect_initial_opinions(query, experts)
    
    def _interdisciplinary_discussion_phase(self, query: MedicalQuery, experts: List[Dict], previous_responses: List[AgentResponse]) -> List[AgentResponse]:
        """跨学科讨论阶段"""
        return self._conduct_expert_discussion(query, experts, previous_responses)
    
    def _coordinator_synthesis_phase(self, query: MedicalQuery, experts: List[Dict], all_responses: List[AgentResponse]) -> AgentResponse:
        """协调员综合阶段"""
        coordinator = Agent(
            instruction="You are a senior medical coordinator responsible for synthesizing multidisciplinary team discussions.",
            role="MDT Coordinator",
            model_info=self.model_name
        )
        
        all_opinions = "\n\n".join([
            f"**{resp.role}:**\n{resp.response}"
            for resp in all_responses
        ])
        
        coordinator_prompt = f"""
        Medical Query: {query.query_text}
        
        Multidisciplinary Team Analysis:
        {all_opinions}
        
        As the MDT coordinator, provide a final coordinated recommendation that integrates all perspectives (limit to 200 words):
        """
        
        start_time = time.time()
        response_text = coordinator.chat(coordinator_prompt)
        processing_time = time.time() - start_time
        
        return AgentResponse(
            agent_id="mdt_coordinator",
            role="MDT Coordinator",
            response=response_text,
            confidence=0.9,
            tokens_used={"input": coordinator.total_input_tokens, "output": coordinator.total_output_tokens},
            processing_time=processing_time
        )
    
    async def process_query(self, query: MedicalQuery) -> Dict[str, Any]:
        """处理医学查询的主要入口点"""
        print(f"\n[LangGraph] 🚀 开始处理医学查询: {query.query_text[:50]}...")
        
        # 初始化状态
        initial_state = MedicalWorkflowState(
            query=query,
            difficulty_level=None,
            recruited_experts=[],
            agent_responses=[],
            interaction_history=[],
            final_decision=None,
            confidence_score=None,
            processing_metadata=None,
            total_input_tokens=0,
            total_output_tokens=0
        )
        
        # 运行工作流
        start_time = time.time()
        final_state = self.workflow.invoke(initial_state)
        total_time = time.time() - start_time
        
        # 构建返回结果
        result = {
            "query_id": f"medq_{int(time.time())}",
            "query": query.query_text,
            "difficulty_level": final_state["difficulty_level"].value,
            "final_decision": final_state["final_decision"],
            "confidence_score": final_state["confidence_score"],
            "expert_count": len(final_state["agent_responses"]),
            "token_usage": {
                "input_tokens": final_state["total_input_tokens"],
                "output_tokens": final_state["total_output_tokens"],
                "total_tokens": final_state["total_input_tokens"] + final_state["total_output_tokens"]
            },
            "processing_time": total_time,
            "metadata": final_state["processing_metadata"]
        }
        
        print(f"\n[LangGraph] ✨ 查询处理完成! 耗时: {total_time:.2f}s")
        print(f"[LangGraph] 📊 Token使用: {result['token_usage']['total_tokens']}")
        print(f"[LangGraph] 🎯 置信度: {result['confidence_score']:.2f}")
        
        return result

# 使用示例和测试代码
if __name__ == "__main__":
    # 示例使用
    mediagents = MediAgentsLangGraph()
    
    # 测试查询
    test_query = MedicalQuery(
        query_text="A 45-year-old patient presents with chest pain and shortness of breath. What should be the immediate diagnostic approach?",
        query_type=QueryType.DIAGNOSIS
    )
    
    # 处理查询
    result = asyncio.run(mediagents.process_query(test_query))
    
    # 输出结果
    print("\n" + "="*60)
    print("🏥 MEDLANGRAPH 处理结果")
    print("="*60)
    print(f"查询ID: {result['query_id']}")
    print(f"难度级别: {result['difficulty_level']}")
    print(f"专家数量: {result['expert_count']}")
    print(f"处理时间: {result['processing_time']:.2f}秒")
    print(f"置信度: {result['confidence_score']:.2f}")
    print(f"Token使用: {result['token_usage']['total_tokens']}")
    print("\n💡 最终建议:")
    print("-" * 40)
    print(result['final_decision'])
    print("="*60)