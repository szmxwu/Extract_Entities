"""
实体合并模块
用于合并同一解剖分支上的父子节点实体
"""

from typing import List, Dict, Any, Tuple, Set
import re
from collections import defaultdict
import copy


class EntityMerger:
    """实体合并器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化合并器
        
        Args:
            config: 配置字典，包含合并规则
        """
        self.config = config or {}
        # 特殊分隔符，用于判断是否需要特殊处理
        self.special_separators = r'[\\|/|+]'
    
    def merge_entities(self, entities: List[Dict[str, Any]], 
                       title_mode: bool = False) -> List[Dict[str, Any]]:
        """
        合并同一分支上的实体
        
        Args:
            entities: 实体列表，每个实体包含short_sentence字段
            title_mode: 是否为标题模式
        
        Returns:
            合并后的实体列表
        """
        if not entities:
            return entities
        
        # 去重：移除完全相同的实体
        entities = self._remove_duplicates(entities)
        
        # 按句子分组
        sentence_groups = self._group_by_sentence(entities)
        
        # 处理每个句子的实体
        merged_entities = []
        for sentence_key, sentence_entities in sentence_groups.items():
            if len(sentence_entities) == 1:
                # 只有一个实体，直接添加
                merged_entities.append(sentence_entities[0])
            else:
                # 多个实体，进行合并处理
                merged = self._merge_sentence_entities(
                    sentence_entities, 
                    title_mode
                )
                merged_entities.extend(merged)
        
        # 按原始顺序排序
        merged_entities = sorted(merged_entities, 
                                key=lambda x: (x.get('sentence_index', 0), 
                                             x.get('sentence_start', 0),
                                             x.get('start', 0)))
        
        return merged_entities
    
    def _remove_duplicates(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        移除完全相同的实体
        
        Args:
            entities: 实体列表
        
        Returns:
            去重后的实体列表
        """
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # 创建实体的唯一标识
            entity_key = self._create_entity_key(entity)
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _create_entity_key(self, entity: Dict[str, Any]) -> str:
        """
        创建实体的唯一标识键
        
        Args:
            entity: 实体字典
        
        Returns:
            唯一标识字符串
        """
        # 使用关键字段创建唯一标识，包括sentence_start
        key_fields = [
            str(entity.get('sentence_index', '')),
            str(entity.get('sentence_start', '')),  # 使用sentence_start
            str(entity.get('start', '')),
            entity.get('keyword', ''),
            str(entity.get('partlist', [])),
            entity.get('orientation', '')
        ]
        return '|'.join(key_fields)
    
    def _group_by_sentence(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        按句子对实体分组
        
        Args:
            entities: 实体列表
        
        Returns:
            分组后的字典
        """
        groups = defaultdict(list)
        
        for entity in entities:
            # 使用句子索引和sentence_start作为分组键
            sentence_index = entity.get('sentence_index', 0)
            sentence_start = entity.get('sentence_start', 0)  # 使用sentence_start
            group_key = f"{sentence_index}_{sentence_start}"
            groups[group_key].append(entity)
        
        return dict(groups)
    
    def _merge_sentence_entities(self, entities: List[Dict[str, Any]], 
                                title_mode: bool) -> List[Dict[str, Any]]:
        """
        合并同一句子中的实体
        
        Args:
            entities: 同一句子的实体列表
            title_mode: 是否为标题模式
        
        Returns:
            合并后的实体列表
        """
        # 标记每个实体是否需要合并
        for entity in entities:
            entity['_merge'] = False
        
        # 首先处理keyword包含关系
        entities = self._merge_keyword_containment(entities)
        
        if title_mode:
            # 标题模式：优先保留父节点
            merged = self._merge_title_mode(entities)
        else:
            # 报告模式：优先保留子节点（更具体）
            merged = self._merge_report_mode(entities)
        
        # 返回未被合并的实体
        result = [e for e in merged if not e.get('_merge', False)]
        
        # 清理临时标记
        for entity in result:
            entity.pop('_merge', None)
        
        return result
    
    def _merge_keyword_containment(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理keyword包含关系：如果两个实体的keyword存在包含关系，保留较长的那个
        
        Args:
            entities: 实体列表
            
        Returns:
            处理后的实体列表
        """
        # 按start位置排序，便于处理
        sorted_entities = sorted(entities, key=lambda x: x.get('start', 0))
        
        for i in range(len(sorted_entities)):
            if sorted_entities[i]['_merge']:
                continue
                
            entity_i = sorted_entities[i]
            start_i = entity_i.get('start', 0)
            end_i = entity_i.get('end', 0)
            keyword_i = entity_i.get('keyword', '')
            
            for j in range(i + 1, len(sorted_entities)):
                if sorted_entities[j]['_merge']:
                    continue
                    
                entity_j = sorted_entities[j]
                start_j = entity_j.get('start', 0)
                end_j = entity_j.get('end', 0)
                keyword_j = entity_j.get('keyword', '')
                
                # 检查是否存在包含关系
                is_i_contains_j = start_i <= start_j and end_i >= end_j
                is_j_contains_i = start_j <= start_i and end_j >= end_i
                
                if is_i_contains_j and not is_j_contains_i:
                    # i包含j，保留i（较长），删除j
                    sorted_entities[j]['_merge'] = True
                elif is_j_contains_i and not is_i_contains_j:
                    # j包含i，保留j（较长），删除i
                    sorted_entities[i]['_merge'] = True
                    break  # i已被标记删除，不需要继续比较
                # 如果完全相等或者没有包含关系，则不处理
                
        return sorted_entities
    
    def _merge_report_mode(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        报告模式合并：保留更具体的子节点
        
        Args:
            entities: 实体列表
        
        Returns:
            合并后的实体列表
        """
        # 按partlist长度降序排序（先处理更具体的实体）
        sorted_entities = sorted(entities, 
                                key=lambda x: len(self._get_partlist(x)), 
                                reverse=True)
        
        for i in range(len(sorted_entities) - 1):
            if sorted_entities[i]['_merge']:
                continue
            
            entity_i = sorted_entities[i]
            partlist_i = self._get_partlist(entity_i)
            
            for j in range(i + 1, len(sorted_entities)):
                if sorted_entities[j]['_merge']:
                    continue
                
                entity_j = sorted_entities[j]
                partlist_j = self._get_partlist(entity_j)
                
                # 检查是否可以合并
                if self._can_merge_report(entity_i, partlist_i, 
                                         entity_j, partlist_j):
                    # 标记较泛的实体为需要合并（删除）
                    sorted_entities[j]['_merge'] = True
        
        return sorted_entities
    
    def _merge_title_mode(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        标题模式合并：根据分隔符决定合并方向
        
        Args:
            entities: 实体列表
        
        Returns:
            合并后的实体列表
        """
        # 按partlist长度升序排序（先处理较泛的实体）
        sorted_entities = sorted(entities, 
                                key=lambda x: len(self._get_partlist(x)))
        
        for i in range(len(sorted_entities) - 1):
            if sorted_entities[i]['_merge']:
                continue
            
            entity_i = sorted_entities[i]
            partlist_i = self._get_partlist(entity_i)
            
            for j in range(i + 1, len(sorted_entities)):
                if sorted_entities[j]['_merge']:
                    continue
                
                entity_j = sorted_entities[j]
                partlist_j = self._get_partlist(entity_j)
                
                # 使用short_sentence获取两个实体之间的文本
                short_sentence = entity_i.get('short_sentence', '')
                if short_sentence:
                    # 获取两个实体在句子中的位置
                    start_i = entity_i.get('end', entity_i.get('start', 0))
                    start_j = entity_j.get('start', 0)
                    
                    # 如果位置合理，提取中间文本
                    if 0 <= start_i < len(short_sentence) and start_i < start_j <= len(short_sentence):
                        between_text = short_sentence[start_i:start_j]
                    else:
                        between_text = ""
                else:
                    between_text = ""
                
                # 根据中间文本决定合并策略
                if re.search(self.special_separators, between_text):
                    # 有特殊分隔符，保留子节点
                    if self._is_parent_child(partlist_i, partlist_j):
                        sorted_entities[j]['_merge'] = True
                else:
                    # 无特殊分隔符，保留父节点
                    if self._is_parent_child(partlist_j, partlist_i):
                        sorted_entities[i]['_merge'] = True
                        break
        
        return sorted_entities
    
    def _can_merge_report(self, entity1: Dict[str, Any], partlist1: List,
                         entity2: Dict[str, Any], partlist2: List) -> bool:
        """
        判断报告模式下两个实体是否可以合并
        
        Args:
            entity1: 第一个实体（更具体）
            partlist1: 第一个实体的partlist
            entity2: 第二个实体（更泛）
            partlist2: 第二个实体的partlist
        
        Returns:
            是否可以合并
        """
        # 检查是否为父子关系
        if not self._is_parent_child(partlist1, partlist2):
            return False
        
        # 检查方位是否兼容
        orient1 = entity1.get('orientation', '')
        orient2 = entity2.get('orientation', '')
        if not self._is_orientation_compatible(orient1, orient2):
            return False
        
        #检查illness是否为空
        illness1 = entity1.get('illness', '')
        illness2 = entity2.get('illness', '')
        if not illness1 or not illness2:
            return True
        
        # 检查阳性状态是否相同
        positive1 = entity1.get('positive', False)
        positive2 = entity2.get('positive', False)
        if positive1 != positive2 :
            return False
        
        return True
    
    def _is_parent_child(self, child_partlist: List, parent_partlist: List) -> bool:
        """
        判断是否为父子关系（child包含parent的所有元素）
        
        Args:
            child_partlist: 可能的子节点partlist
            parent_partlist: 可能的父节点partlist
        
        Returns:
            是否为父子关系
        """
        # 处理嵌套列表的情况
        child_list = self._flatten_partlist(child_partlist)
        parent_list = self._flatten_partlist(parent_partlist)
        
        # 子节点应该包含父节点的所有元素
        return set(parent_list).issubset(set(child_list)) or set(child_list).issubset(set(parent_list))
    
    def _is_orientation_compatible(self, orient1: str, orient2: str) -> bool:
        """
        判断两个方位是否兼容
        
        Args:
            orient1: 方位1
            orient2: 方位2
        
        Returns:
            是否兼容
        """
        # 如果有一个为空，则兼容
        if not orient1 or not orient2:
            return True
        
        # 否则必须相同
        return orient1 == orient2
    
    def _get_partlist(self, entity: Dict[str, Any]) -> List:
        """
        获取实体的partlist（处理嵌套情况）
        
        Args:
            entity: 实体字典
        
        Returns:
            partlist列表
        """
        partlist = entity.get('partlist', [])
        
        # 如果是嵌套列表，取第一个
        if partlist and isinstance(partlist[0], tuple):
            return list(partlist[0])
        
        return partlist
    
    def _flatten_partlist(self, partlist: List) -> List:
        """
        展平partlist（处理可能的嵌套）
        
        Args:
            partlist: 可能嵌套的partlist
        
        Returns:
            展平后的列表
        """
        if not partlist:
            return []
        
        # 如果第一个元素是列表，说明是嵌套的
        if isinstance(partlist[0], tuple) or isinstance(partlist[0], list):
            return list(partlist[0])
        
        return partlist
    
    def get_merge_statistics(self, original: List, merged: List) -> Dict[str, int]:
        """
        获取合并统计信息
        
        Args:
            original: 原始实体列表
            merged: 合并后的实体列表
        
        Returns:
            统计信息字典
        """
        return {
            'original_count': len(original),
            'merged_count': len(merged),
            'reduced_count': len(original) - len(merged),
            'reduction_rate': (len(original) - len(merged)) / len(original) * 100 
                            if original else 0
        }



def merge_part(data_dict: List[Dict[str, Any]], 
              title: bool = False) -> List[Dict[str, Any]]:
    """
    合并同一分支上的实体（兼容原有接口）
    
    Args:
        data_dict: 实体字典列表
        pre_ReportStr: （已废弃）预处理后的文本，现在从实体的short_sentence获取
        title: 是否为标题模式
    
    Returns:
        合并后的实体列表
    """
    merger = EntityMerger()
    return merger.merge_entities(data_dict, title)


# 测试代码
if __name__ == "__main__":
    import json
    
    # 测试包含关系的数据
    test_containment = [
        {
            "keyword": "L5/S1",
            "partlist": [["脊柱", "L5/S1"]],
            "axis": [[106.0, 107.0]],
            "position": "L5/S1",
            "orientation": "",
            "sentence_index": 0,
            "sentence_start": 0,
            "start": 0,
            "end": 5,
            "short_sentence": "L5/S1椎间盘突出",
            "long_sentence": "L5/S1椎间盘突出",
            "positive": False
        },
        {
            "keyword": "L5",
            "partlist": [["脊柱", "L5"]],
            "axis": [[106.0, 107.0]],
            "position": "L5",
            "orientation": "",
            "sentence_index": 0,
            "sentence_start": 0,
            "start": 0,
            "end": 2,
            "short_sentence": "L5/S1椎间盘突出",
            "long_sentence": "L5/S1椎间盘突出",
            "positive": False
        }
    ]
    
    print("=== 测试keyword包含关系处理 ===")
    merger = EntityMerger()
    result_containment = merger.merge_entities(test_containment, False)
    
    print("包含关系测试 - 合并前实体数量:", len(test_containment))
    print("包含关系测试 - 合并后实体数量:", len(result_containment))
    print("\n包含关系测试 - 合并后的实体:")
    for entity in result_containment:
        print(f"  - {entity['keyword']} (start:{entity['start']}, end:{entity['end']})")
    
    # 原有测试数据（包含必要的字段）
    test_entities = [
        {
            "keyword": "肝脏",
            "partlist": [["腹部", "肝脏"]],
            "axis": [[106.0, 107.0]],
            "position": "肝脏",
            "orientation": "",
            "sentence_index": 0,
            "sentence_start": 0,  # short_sentence在long_sentence中的起始位置
            "start": 0,  # 实体在short_sentence中的起始位置
            "end": 2,
            "short_sentence": "肝脏增大，肝门区可见肿块",
            "long_sentence": "肝脏增大，肝门区可见肿块，胆囊未见异常",
            "positive": False
        },
        {
            "keyword": "肝门",
            "partlist": [["腹部", "肝脏", "肝门"]],
            "axis": [[106.0, 107.0]],
            "position": "肝门",
            "orientation": "",
            "sentence_index": 0,
            "sentence_start": 0,
            "start": 5,  # 在short_sentence中的位置
            "end": 7,
            "short_sentence": "肝脏增大，肝门区可见肿块",
            "long_sentence": "肝脏增大，肝门区可见肿块，胆囊未见异常",
            "positive": False
        },
        {
            "keyword": "胆囊",
            "partlist": [["腹部", "胆囊"]],
            "axis": [[106.0, 107.0]],
            "position": "胆囊",
            "orientation": "",
            "sentence_index": 0,
            "sentence_start": 13,  # 新句子在long_sentence中的起始位置
            "start": 0,
            "end": 2,
            "short_sentence": "胆囊未见异常",
            "long_sentence": "肝脏增大，肝门区可见肿块，胆囊未见异常",
            "positive": True
        }
    ]
    
    print("\n=== 原有功能测试 ===")
    # 执行合并
    result = merger.merge_entities(test_entities, False)
    
    # 打印结果
    print("合并前实体数量:", len(test_entities))
    print("合并后实体数量:", len(result))
    print("\n合并后的实体:")
    for entity in result:
        print(f"  - {entity['keyword']} ({entity['position']}) "
              f"sentence_start={entity['sentence_start']}")
    
    # 获取统计信息
    stats = merger.get_merge_statistics(test_entities, result)
    print(f"\n合并统计: 减少了 {stats['reduced_count']} 个实体 "
          f"({stats['reduction_rate']:.1f}%)")