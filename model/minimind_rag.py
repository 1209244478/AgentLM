"""
Mini-RAG 适配器：将 Mini-RAG 检索增强生成与 MiniMind 模型集成

提供两种模式：
1. 独立模式：直接使用 Mini-RAG 的 insert/query 接口
2. 增强模式：将检索结果注入 MiniMind 的生成上下文，实现 RAG 增强
"""
import os
import sys
import json
import asyncio

# 添加 mini-RAG 路径
_MINI_RAG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mini-RAG')
if _MINI_RAG_PATH not in sys.path:
    sys.path.insert(0, _MINI_RAG_PATH)

try:
    from minirag import MiniRAG
    from minirag.base import QueryParam
    HAS_MINIRAG = True
except ImportError:
    HAS_MINIRAG = False


class MiniMindRAG:
    """MiniMind 的 RAG 增强适配器

    将 Mini-RAG 的异构图检索能力与 MiniMind 模型的生成能力结合，
    在推理时自动检索相关文档片段并注入到上下文中。

    用法:
        rag = MiniMindRAG(working_dir="./rag_cache")
        rag.insert("文档内容...")
        result = rag.query_with_model("用户问题", model, tokenizer)
    """

    def __init__(self, working_dir="./rag_cache", embedding_func=None, llm_func=None, **kwargs):
        if not HAS_MINIRAG:
            raise ImportError(
                "Mini-RAG 未安装。请确保 mini-RAG 目录存在于项目根目录下，"
                "并安装其依赖：pip install -r mini-RAG/requirements.txt"
            )
        self.working_dir = working_dir
        self.rag = MiniRAG(
            working_dir=working_dir,
            embedding_func=embedding_func,
            llm_model_func=llm_func,
            **kwargs
        )

    def insert(self, text_or_texts):
        """插入文档到知识库"""
        return self.rag.insert(text_or_texts)

    def insert_file(self, file_path):
        """从文件插入文档"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.insert(content)

    def query(self, question, mode="mini", only_need_context=False):
        """查询知识库

        Args:
            question: 查询问题
            mode: 检索模式 ("mini"/"light"/"naive")
            only_need_context: 仅返回检索到的上下文，不生成回答
        Returns:
            检索结果或生成的回答
        """
        param = QueryParam(mode=mode, only_need_context=only_need_context)
        return self.rag.query(question, param)

    def query_context(self, question, mode="mini", top_k=3):
        """仅获取检索上下文，用于注入到 MiniMind 生成流程

        Args:
            question: 查询问题
            mode: 检索模式
            top_k: 返回 top_k 个相关片段
        Returns:
            str: 检索到的相关文本片段
        """
        try:
            param = QueryParam(mode=mode, only_need_context=True)
            context = self.rag.query(question, param)
            return context if isinstance(context, str) else str(context)
        except Exception:
            return ""

    def _build_rag_messages(self, question, context, max_context_tokens=2048, tokenizer=None):
        """构建 RAG 增强的消息，带上下文长度管理

        Args:
            question: 用户问题
            context: 检索到的上下文
            max_context_tokens: 上下文最大 token 数
            tokenizer: 分词器（用于精确截断）
        Returns:
            messages: 构建好的消息列表
        """
        if context:
            # 截断过长的上下文
            if tokenizer is not None:
                context_tokens = tokenizer.encode(context, add_special_tokens=False)
                if len(context_tokens) > max_context_tokens:
                    context_tokens = context_tokens[:max_context_tokens]
                    context = tokenizer.decode(context_tokens, skip_special_tokens=True)
                    context += "\n...(内容过长已截断)"
            elif len(context) > max_context_tokens * 4:  # 粗略估计：1 token ≈ 4 chars
                context = context[:max_context_tokens * 4] + "\n...(内容过长已截断)"

            messages = [
                {"role": "system", "content": "你是一个知识丰富的AI助手。请根据以下参考资料回答用户的问题。如果参考资料中没有相关信息，请诚实地说明你不知道。\n\n参考资料：\n" + context},
                {"role": "user", "content": question}
            ]
        else:
            messages = [
                {"role": "system", "content": "你是一个知识丰富的AI助手。如果你不确定答案，请诚实地说明。"},
                {"role": "user", "content": question}
            ]
        return messages

    def query_with_model(self, question, model, tokenizer, mode="mini", max_new_tokens=512, max_context_tokens=2048, **generate_kwargs):
        """RAG 增强的模型生成

        1. 先通过 Mini-RAG 检索相关上下文
        2. 将上下文注入到 prompt 中（带长度管理）
        3. 用 MiniMind 模型生成回答

        Args:
            question: 用户问题
            model: MiniMindForCausalLM 实例
            tokenizer: 分词器
            mode: RAG 检索模式
            max_new_tokens: 最大生成 token 数
            max_context_tokens: 检索上下文最大 token 数
            **generate_kwargs: 传递给 model.generate() 的额外参数
        Returns:
            str: 模型生成的回答
        """
        # Step 1: 检索上下文
        context = self.query_context(question, mode=mode)

        # Step 2: 构建 RAG 增强的 prompt（带长度管理）
        messages = self._build_rag_messages(question, context, max_context_tokens, tokenizer)

        # Step 3: 用模型生成
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 检查总长度是否超过模型最大位置编码
        max_pos = getattr(model.config, 'max_position_embeddings', 32768)
        if inputs["input_ids"].shape[1] + max_new_tokens > max_pos:
            # 截断输入以留出生成空间
            max_input_len = max_pos - max_new_tokens
            if max_input_len > 0:
                inputs["input_ids"] = inputs["input_ids"][:, :max_input_len]
                inputs["attention_mask"] = inputs["attention_mask"][:, :max_input_len]

        with __import__('torch').no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                **generate_kwargs
            )

        # 解码，只取生成部分
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def query_with_model_ttt(self, question, model, tokenizer, mode="mini",
                              max_new_tokens=512, ttt_lr=1e-4, ttt_interval=64,
                              max_context_tokens=2048, **generate_kwargs):
        """RAG + TTT 混合增强的模型生成

        在 RAG 检索的基础上，同时启用 In-Place TTT 进行推理时权重更新，
        让模型在生成过程中适应检索到的上下文。

        Args:
            question: 用户问题
            model: MiniMindForCausalLM 实例
            tokenizer: 分词器
            mode: RAG 检索模式
            max_new_tokens: 最大生成 token 数
            ttt_lr: TTT 学习率
            ttt_interval: TTT 更新间隔
            max_context_tokens: 检索上下文最大 token 数
            **generate_kwargs: 传递给 model.generate() 的额外参数
        Returns:
            str: 模型生成的回答
        """
        # 启用 TTT
        model.enable_ttt(lr=ttt_lr)
        ttt_enabled = True

        try:
            # 检索上下文
            context = self.query_context(question, mode=mode)

            # 构建 prompt（带长度管理）
            messages = self._build_rag_messages(question, context, max_context_tokens, tokenizer)

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # 检查总长度
            max_pos = getattr(model.config, 'max_position_embeddings', 32768)
            if inputs["input_ids"].shape[1] + max_new_tokens > max_pos:
                max_input_len = max_pos - max_new_tokens
                if max_input_len > 0:
                    inputs["input_ids"] = inputs["input_ids"][:, :max_input_len]
                    inputs["attention_mask"] = inputs["attention_mask"][:, :max_input_len]

            with __import__('torch').no_grad():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    use_ttt=True,
                    ttt_interval=ttt_interval,
                    **generate_kwargs
                )

            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as e:
            # 生成失败时确保 TTT 权重恢复
            raise e
        finally:
            # 禁用 TTT，恢复初始权重（带健壮性保障）
            if ttt_enabled:
                try:
                    model.disable_ttt()
                except Exception:
                    # 如果 disable_ttt 也失败（如权重已被部分修改），
                    # 尝试 reset_ttt_weights 作为后备
                    try:
                        model.reset_ttt_weights()
                    except Exception:
                        pass  # 最后手段：权重可能不一致，但不阻塞调用方
