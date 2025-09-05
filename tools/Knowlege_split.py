# -*- coding: utf-8 -*-
import pandas as pd
import os

def build_tree_from_excel(df: pd.DataFrame) -> dict:
    """
    从Excel DataFrame构建一个嵌套的字典树来表示知识图谱层级。
    """
    tree = {}
    # 注意：这里的层级从二级开始，因为一级部位已经是筛选条件
    level_columns = ['二级部位', '三级部位', '四级部位', '五级部位', '六级部位']
    
    for _, row in df.iterrows():
        # 根节点是当前正在处理的一级部位
        root_name = row['一级部位'].split('|')[0]
        if root_name not in tree:
            tree[root_name] = {}
            
        current_level_dict = tree[root_name]
        
        for col in level_columns:
            part_name = row[col]
            if pd.isna(part_name):
                break
            
            primary_name = part_name.split('|')[0]
            
            if primary_name not in current_level_dict:
                current_level_dict[primary_name] = {}
            current_level_dict = current_level_dict[primary_name]
            
    return tree

def generate_text_from_tree(node: dict, indent_level: int = 0) -> str:
    """
    通过递归遍历树，生成Markdown风格的嵌套列表文本。
    """
    text_representation = ""
    indent = "  " * indent_level
    
    for part_name, children in node.items():
        text_representation += f"{indent}- {part_name}\n"
        if children:
            text_representation += generate_text_from_tree(children, indent_level + 1)
            
    return text_representation

def main(excel_path: str, output_dir: str):
    """
    主函数，读取Excel，按一级部位拆分知识图谱，并保存为多个txt文件。
    """
    print(f"正在从 '{excel_path}' 读取知识图谱...")
    
    try:
        kg_df = pd.read_excel(excel_path, sheet_name=0)
    except FileNotFoundError:
        print(f"错误：文件 '{excel_path}' 未找到。请确保文件路径正确。")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    print(f"知识图谱分片文件将保存至: '{output_dir}' 目录")

    # 兼容同义词，并处理可能的空值
    kg_df = kg_df.dropna(subset=['一级部位'])
    kg_df['一级部位_clean'] = kg_df['一级部位'].astype(str).str.split('|').str[0]
    
    # 获取所有唯一的一级部位
    top_level_parts = kg_df['一级部位_clean'].unique()
    
    print("\n--- 开始处理每个一级部位 ---")
    summary_report = []

    for part_name in top_level_parts:
        # 1. 筛选出属于当前一级部位的子DataFrame
        subset_df = kg_df[kg_df['一级部位_clean'] == part_name].copy()
        
        # 2. 从子DataFrame构建该部位的层级树
        knowledge_tree = build_tree_from_excel(subset_df)
        
        # 3. 从树生成文本表示
        knowledge_text = generate_text_from_tree(knowledge_tree)
        
        # 4. 计算字符数
        char_count = len(knowledge_text)
        
        # 5. 定义输出文件名并保存文件
        # 清理文件名，避免特殊字符导致保存失败
        safe_part_name = "".join(c for c in part_name if c.isalnum() or c in (' ', '_')).rstrip()
        output_filename = os.path.join(output_dir, f"kg_{safe_part_name}.txt")
        
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(knowledge_text)
            
        summary_report.append({
            "part_name": part_name,
            "filename": output_filename,
            "char_count": char_count
        })

    # 6. 打印总结报告
    print("\n--- 知识图谱拆分完成总结报告 ---\n")
    print(f"{'一级部位':<15} {'生成文件名':<30} {'字符数':<10}")
    print("-" * 60)
    for report in summary_report:
        print(f"{report['part_name']:<15} {report['filename']:<30} {report['char_count']:<10}")
    print("-" * 60)
    print(f"\n总计生成 {len(summary_report)} 个知识图谱分片文件。")


if __name__ == "__main__":
    # 请将 '报告助手部位词典.xlsx' 替换为您的实际文件路径
    excel_file_path = "报告助手部位词典.xlsx"
    # 定义输出目录
    output_directory = "kg_splits"
    
    # 运行前请确保已安装所需库:
    # pip install pandas openpyxl
    main(excel_file_path, output_directory)

