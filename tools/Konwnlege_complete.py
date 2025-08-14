# -*- coding: utf-8 -*-
import pandas as pd
import requests
import json
import time
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from json_repair import repair_json
import re
from dotenv import load_dotenv
import os


# 从.env文件加载环境变量
load_dotenv()

# --- 1. 本地LLM配置 (从环境变量读取) ---
LOCAL_LLM_API_ENDPOINT = os.getenv("LOCAL_LLM_API_ENDPOINT", "YOUR_LOCAL_LLM_API_ENDPOINT")
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "YOUR_MODEL_NAME")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")

# --- 2. 核心函数 ---

def extract_json_from_markdown(text: str) -> str:
    """
    从可能包含Markdown代码块标记的文本中提取JSON内容
    """
    # 去掉可能的Markdown代码块标记
    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    return text

def parse_llm_response(response_text: str) -> str:
    """
    解析和优化大模型的响应输出
    """
    # 去除首尾空白字符
    response_text=re.sub(r'<think>.*?</think>', '', response_text,flags=re.DOTALL)

    response_text = response_text.replace("\n","").strip()
    
    # 去掉Markdown代码块标记
    response_text = extract_json_from_markdown(response_text)
    
    # 尝试修复JSON格式
    try:
        # 如果是JSON格式，尝试修复并提取标签
        repaired_json = repair_json(response_text)
        if repaired_json:
            parsed_data = json.loads(repaired_json)
            if isinstance(parsed_data, dict) and '标签' in parsed_data:
                return parsed_data['标签']
            elif isinstance(parsed_data, str):
                return parsed_data
    except:
        pass
    
    # 如果不是JSON或无法修复，直接处理文本
    # 去掉可能的引号
    response_text = response_text.strip().strip('"').strip("'")
    
    # 如果有明确的"输出:"标记，只取后面的部分
    if '输出:' in response_text:
        response_text = response_text.split('输出:')[-1].strip()
    
    return response_text



# --- 2. 核心函数 ---

def get_enriched_attributes_from_llm(entity_name: str, full_path: list) -> Dict[str, Any]:
    """
    调用本地LLM，为一个实体节点生成一套丰富的结构化属性。
    采用“思维链”提示词，引导LLM进行分步思考。
    """
    headers = {
        "Content-Type": "application/json",
    }
    if LOCAL_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LOCAL_LLM_API_KEY}"

    prompt = f"""
# 角色
你是一位顶级的医学知识图谱构建专家，负责对【解剖学实体】进行深度分析和结构化。/no_think

# 任务
分析给定的【解剖学实体】及其在知识图谱中的【上下文路径】，然后输出一个包含其所有结构化属性的JSON对象。

# 知识图谱Schema定义
1.  EntityType: 实体类型。可选值为：
    - `AnatomicalEntity`: 一个标准的、独立的解剖结构。
    - `RelationalSpace`: 一个由两个或多个实体共同定义的关系、空间或间隙。
    - `FunctionalSystem`: 一个功能性的系统或集合，而非具体的解剖位置。
    - `AbstractConcept`: 一个抽象概念，如“病灶”、“占位”。
2.  RelatedParts: 与该实体紧密相关的其他实体（仅当EntityType为RelationalSpace时填写）。
3.  FunctionalSystemTags: 该实体所属的一个或多个功能系统。

# 现有知识图谱的上下文路径
{json.dumps(full_path, ensure_ascii=False)}

# 新任务
---
**解剖学实体**: "{entity_name}"
---

请遵循以下【思考步骤】进行分析，并最后只生成最终的JSON输出：
1.  **实体定性**: 判断【解剖学实体】的`EntityType`是什么？例如，“舟月关节间隙”的本质是由“舟骨”和“月骨”定义的关系空间。
2.  **功能归属**: 判断该实体属于哪些功能系统？例如，“肾上腺”同时属于“泌尿系统”和“内分泌系统”。
3.  **关联分析**: 如果它是一个`RelationalSpace`，它关联了哪些核心实体？
4.  **形成JSON**: 根据以上分析，严格按照指定的格式生成JSON对象。

# 输出格式 (JSON)
{{
  "entity_type": "...",
  "related_parts": ["...", "..."],
  "functional_system_tags": ["...", "..."],
  "explanation": "简要解释你做出此判断的理由。"
}}
"""

    payload = {
        "model": LOCAL_LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
        "response_format": {"type": "json_object"} # 尝试使用JSON模式
    }

    try:
        response = requests.post(
            LOCAL_LLM_API_ENDPOINT,
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        response.raise_for_status()
        response_data = response.json()
        
        # 提取核心内容和token消耗
        result_json_str = response_data['choices'][0]['message']['content']
        usage = response_data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        
        # 解析LLM返回的JSON字符串
        result_json_str=parse_llm_response(result_json_str)
        enriched_data = json.loads(result_json_str)
        enriched_data['prompt_tokens'] = prompt_tokens
        enriched_data['completion_tokens'] = completion_tokens
        
        return enriched_data

    except Exception as e:
        print(f"\n错误: 实体 '{entity_name}' 处理失败 - {e}")
        return {
            "entity_type": "ERROR", "related_parts": [], 
            "functional_system_tags": [], "explanation": str(e),
            "prompt_tokens": 0, "completion_tokens": 0
        }


def build_unique_node_tree(df: pd.DataFrame) -> Dict[str, Any]:
    """
    从Excel DataFrame构建一个唯一的、去重的嵌套字典树。
    """
    tree = {}
    level_columns = ['一级部位', '二级部位', '三级部位', '四级部位', '五级部位', '六级部位']
    for _, row in df.iterrows():
        current_level_dict = tree
        for col in level_columns:
            part_name = row[col]
            if pd.isna(part_name): break
            primary_name = str(part_name).split('|')[0]
            if primary_name not in current_level_dict:
                current_level_dict[primary_name] = {}
            current_level_dict = current_level_dict[primary_name]
    return tree

def get_all_unique_paths(node: Dict, path: List, all_paths: List):
    """
    递归函数，用于从树中提取所有唯一的节点路径。
    """
    if not node:
        return
    for key, children in node.items():
        current_path = path + [key]
        all_paths.append(tuple(current_path))
        get_all_unique_paths(children, current_path, all_paths)

# --- 3. 主处理脚本 ---
def main(excel_path: str, output_path: str):
    """
    主函数，实现对现有知识图谱的全面改造升级，并报告成本。
    """
    print(f"正在从 '{excel_path}' 读取知识图谱...")
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
    except FileNotFoundError:
        print(f"错误：文件 '{excel_path}' 未找到。")
        return

    start_time = time.time()

    # --- 步骤 1: 构建唯一的节点树和路径列表 ---
    print("步骤 1/3: 正在构建唯一的节点树...")
    unique_tree = build_unique_node_tree(df)
    unique_paths = []
    get_all_unique_paths(unique_tree, [], unique_paths)
    unique_nodes = {node for path in unique_paths for node in path}
    print(f"发现 {len(unique_nodes)} 个唯一实体节点需要处理。")

    # --- 步骤 2: 为唯一的节点调用LLM生成丰富的属性 ---
    print("步骤 2/3: 正在为唯一节点调用LLM生成丰富的属性...")
    enriched_node_data = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # 路径到节点的映射，便于查找上下文
    path_map = {path[-1]: list(path) for path in sorted(unique_paths, key=len)}

    for node_name in tqdm(sorted(list(unique_nodes)), desc="处理唯一节点"):
        context_path = path_map.get(node_name, [node_name])
        attributes = get_enriched_attributes_from_llm(node_name, context_path)
        enriched_node_data[node_name] = attributes
        
        total_prompt_tokens += attributes.get('prompt_tokens', 0)
        total_completion_tokens += attributes.get('completion_tokens', 0)
        time.sleep(0.1)

    # --- 步骤 3: 构建并保存新的知识图谱Excel文件 ---
    print("步骤 3/3: 正在构建新的知识图谱DataFrame并保存...")
    
    new_kg_records = []
    for path in unique_paths:
        record = {}
        for i, node_name in enumerate(path):
            record[f'{i+1}级部位'] = node_name
            # 获取该节点的丰富属性
            node_info = enriched_node_data.get(node_name, {})
            if i == len(path) - 1: # 只为最末端的节点附加完整信息
                record['EntityType'] = node_info.get('entity_type')
                record['RelatedParts'] = '|'.join(node_info.get('related_parts', []))
                record['FunctionalSystemTags'] = ','.join(node_info.get('functional_system_tags', []))
                record['LLM_Explanation'] = node_info.get('explanation')
        new_kg_records.append(record)

    new_df = pd.DataFrame(new_kg_records)
    # 调整列顺序
    cols = [f'{i+1}级部位' for i in range(6)] + ['EntityType', 'RelatedParts', 'FunctionalSystemTags', 'LLM_Explanation']
    new_df = new_df.reindex(columns=cols)
    
    try:
        new_df.to_excel(output_path, index=False)
        print(f"处理完成！新的知识图谱已保存至 '{output_path}'。")
    except Exception as e:
        print(f"错误：保存Excel文件失败 - {e}")

    # --- 4. 打印成本评估报告 ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n--- 知识图谱升级成本评估报告 ---")
    print(f"处理的唯一节点总数: {len(unique_nodes)}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每个节点耗时: {elapsed_time / len(unique_nodes):.2f} 秒" if unique_nodes else "N/A")
    print("-" * 40)
    print(f"总消耗 Prompt Tokens: {total_prompt_tokens}")
    print(f"总消耗 Completion Tokens: {total_completion_tokens}")
    print(f"总消耗 Tokens: {total_prompt_tokens + total_completion_tokens}")
    print("=" * 40)


if __name__ == "__main__":
    input_excel_path = "报告助手部位词典.xlsx"
    output_excel_path = "报告助手部位词典_enriched.xlsx"
    
    if "YOUR_LOCAL_LLM_API_ENDPOINT" in LOCAL_LLM_API_ENDPOINT:
        print("="*60)
        print("!! 警告: 请先在脚本中填写您的本地LLM API配置信息 !!")
    else:
        main(input_excel_path, output_excel_path)

