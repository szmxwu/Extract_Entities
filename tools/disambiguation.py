"""
医学实体歧义消解模块
用于解决FlashText提取后的多义实体问题
包含LRU缓存优化
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from itertools import zip_longest
import copy
import configparser
from pathlib import Path
from functools import lru_cache
import hashlib
import json

# 配置文件加载
BASE_DIR = Path(__file__).resolve().parent.parent
config_path = BASE_DIR / 'config' / 'config.ini'
conf = configparser.ConfigParser()
conf.read(config_path, encoding='utf-8')

# 加载配置的关键词
spine_words = conf.get("clean", "spine")
second_root = conf.get("clean", "second_root")
bilateral_organs = conf.get("orientation", "bilateral_organs")
single_organs = conf.get("orientation", "single_organs")


def _make_hashable(obj):
    """将字典或列表转换为可哈希的格式用于缓存"""
    if isinstance(obj, (list, dict)):
        return hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return obj


class AmbiguityResolver:
    """医学实体歧义消解器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化消歧器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path:
            self.conf = configparser.ConfigParser()
            self.conf.read(config_path, encoding='utf-8')
            self.spine_words = self.conf.get("clean", "spine")
            self.second_root = self.conf.get("clean", "second_root")
        else:
            self.spine_words = spine_words
            self.second_root = second_root
        
        # 初始化缓存的实例方法
        self._init_cached_methods()
    
    def _init_cached_methods(self):
        """初始化缓存方法"""
        # 坐标交叉判断缓存
        self._interval_cross_cache = {}
        
        # 先验知识过滤缓存
        self._prior_knowledge_cache = {}
        
        # 单实体消歧缓存
        self._single_ambiguity_cache = {}
    
    def resolve_entities(self, entities_list: List[Dict[str, Any]], 
                        add_info: Optional[List[Tuple[float, float]]] = None) -> List[Dict[str, Any]]:
        """
        对实体列表进行歧义消解
        
        Args:
            entities_list: 实体列表，包含可能有歧义的实体
            add_info: 检查部位的axis信息列表
        
        Returns:
            消歧后的实体列表，保持原有数据结构
        """
        if not entities_list:
            return entities_list
        
        result = []
        
        for i, entity in enumerate(entities_list):
            # 检查是否存在歧义（partlist或axis有多个元素）
            if isinstance(entity.get('partlist', []), list) and \
               len(entity.get('partlist', [])) > 1 and \
               isinstance(entity['partlist'][0], tuple):
                # 存在歧义，进行消解
                resolved = self._resolve_single_ambiguity(entity, entities_list, i, add_info)
                result.append(resolved)
            else:
                # 无歧义或已经是单一候选，直接添加
                result.append(entity)
        
        return result
    
    @lru_cache(maxsize=256)
    def _resolve_single_ambiguity_cached(self, entity_hash: str, 
                                        context_hash: str,
                                        add_info_hash: str) -> str:
        """缓存版本的单个实体消歧"""
        ambiguous_entity = json.loads(entity_hash)
        context_entities = json.loads(context_hash) if context_hash else []
        add_info = json.loads(add_info_hash) if add_info_hash else None
        
        # 执行消歧逻辑
        result = self._resolve_single_ambiguity_logic(ambiguous_entity, context_entities, add_info)
        return json.dumps(result, ensure_ascii=False)
    
    def _resolve_single_ambiguity(self, ambiguous_entity: Dict[str, Any],
                                 all_entities: List[Dict[str, Any]],
                                 current_index: int,
                                 add_info: Optional[List[Tuple[float, float]]]) -> Dict[str, Any]:
        """
        消解单个歧义实体（带缓存）
        
        Args:
            ambiguous_entity: 歧义实体
            all_entities: 所有实体列表
            current_index: 当前实体在列表中的索引
            add_info: 检查部位信息
        
        Returns:
            消歧后的实体
        """
        # 构建上下文
        context_entities = self._build_context_list(all_entities, current_index)
        
        # 生成缓存键
        entity_hash = json.dumps(ambiguous_entity, sort_keys=True, ensure_ascii=False)
        context_hash = json.dumps(context_entities, sort_keys=True, ensure_ascii=False) if context_entities else ""
        add_info_hash = json.dumps(add_info, sort_keys=True, ensure_ascii=False) if add_info else ""
        
        # 使用缓存
        result_str = self._resolve_single_ambiguity_cached(entity_hash, context_hash, add_info_hash)
        return json.loads(result_str)
    
    def _resolve_single_ambiguity_logic(self, ambiguous_entity: Dict[str, Any],
                                       context_entities: List[Dict[str, Any]],
                                       add_info: Optional[List[Tuple[float, float]]]) -> Dict[str, Any]:
        """
        单个实体消歧的实际逻辑
        
        Args:
            ambiguous_entity: 歧义实体
            context_entities: 上下文实体
            add_info: 检查部位信息
        
        Returns:
            消歧后的实体
        """
        # 准备候选列表
        candidates = self._prepare_candidates(ambiguous_entity)
        
        if len(candidates) == 1:
            return self._merge_candidate_to_entity(ambiguous_entity, candidates[0])
        
        # 策略1：先验知识过滤
        filtered = self._apply_prior_knowledge(candidates, ambiguous_entity.get('short_sentence', ''))
        if filtered and len(filtered) < len(candidates):
            candidates = filtered
            if len(candidates) == 1:
                return self._merge_candidate_to_entity(ambiguous_entity, candidates[0])
        
        # 策略2：检查部位匹配
        if add_info:
            filtered = self._match_exam_part(candidates, add_info)
            if filtered and len(filtered) < len(candidates):
                # 特殊处理：排除明显不相关的部位
                if "脊柱" not in [c['partlist'][0] for c in filtered] and \
                   "皮肤软组织" not in [c['position'] for c in filtered]:
                    candidates = filtered
                    if len(candidates) == 1:
                        return self._merge_candidate_to_entity(ambiguous_entity, candidates[0])
        
        # 策略3：上下文partlist匹配
        if context_entities:
            matched = self._match_context_partlist(candidates, context_entities)
            if matched and len(matched) < len(candidates):
                candidates = matched
                if len(candidates) == 1:
                    return self._merge_candidate_to_entity(ambiguous_entity, candidates[0])
        
        # 策略4：上下文axis匹配
        if context_entities:
            matched = self._match_context_axis(candidates, context_entities)
            if matched and len(matched) < len(candidates):
                candidates = matched
                if len(candidates) == 1:
                    return self._merge_candidate_to_entity(ambiguous_entity, candidates[0])
        
        # 策略5：扩展坐标匹配
        matched = self._match_extended_axis(candidates, context_entities, add_info)
        if matched and len(matched) < len(candidates):
            candidates = matched
            if len(candidates) == 1:
                return self._merge_candidate_to_entity(ambiguous_entity, candidates[0])
        
        # 策略6：兜底策略
        final_candidates = self._apply_fallback_strategy(candidates)
        if len(final_candidates) == 1:
            return self._merge_candidate_to_entity(ambiguous_entity, final_candidates[0])
        else:
            # 如果仍有多个候选，保留所有
            return self._merge_multiple_candidates_to_entity(ambiguous_entity, final_candidates)
    
    def _prepare_candidates(self, ambiguous_entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        准备候选实体列表
        
        Args:
            ambiguous_entity: 歧义实体
        
        Returns:
            候选实体列表
        """
        candidates = []
        partlists = ambiguous_entity.get('partlist', [])
        axes = ambiguous_entity.get('axis', [])
        
        # 确保partlist和axis数量一致
        for i in range(min(len(partlists), len(axes))):
            candidate = {
                'partlist': partlists[i],
                'axis': axes[i],
                'position': partlists[i][-1] if partlists[i] else '',
                'index': i
            }
            candidates.append(candidate)
        
        return candidates
    
    def _merge_candidate_to_entity(self, original_entity: Dict[str, Any],
                                  candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        将选中的单个候选合并回原实体
        
        Args:
            original_entity: 原始实体
            candidate: 选中的单个候选
        
        Returns:
            更新后的实体（partlist和axis仍为嵌套列表格式，但只有一个元素）
        """
        result = copy.deepcopy(original_entity)
        
        # 保持嵌套列表格式，但只有一个元素
        result['partlist'] = [candidate['partlist']]
        result['axis'] = [candidate['axis']]
        result['position'] = candidate['position']
        
        return result
    
    def _merge_multiple_candidates_to_entity(self, original_entity: Dict[str, Any],
                                            candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将多个候选合并回原实体（保留多个候选）
        
        Args:
            original_entity: 原始实体
            candidates: 选中的候选列表
        
        Returns:
            更新后的实体
        """
        result = copy.deepcopy(original_entity)
        
        if candidates:
            # 保留多个候选
            result['partlist'] = [c['partlist'] for c in candidates]
            result['axis'] = [c['axis'] for c in candidates]
            # position取第一个候选的
            result['position'] = candidates[0]['position']
        
        return result
    
    def _build_context_list(self, all_entities: List[Dict[str, Any]], 
                           current_index: int) -> List[Dict[str, Any]]:
        """
        构建上下文实体列表（前后文交替）
        
        Args:
            all_entities: 所有实体
            current_index: 当前索引
        
        Returns:
            上下文实体列表
        """
        all_entities=[x for x in all_entities if len(x['partlist'])==1]
        before = all_entities[:current_index][::-1]  # 反转，从近到远
        after = all_entities[current_index + 1:]
        
        # 交替合并前后文
        context = []
        for b, a in zip_longest(before, after):
            if b is not None:
                context.append(b)
            if a is not None:
                context.append(a)
        
        return context
    
    @lru_cache(maxsize=512)
    def _apply_prior_knowledge_cached(self, candidates_hash: str, sentence_text: str) -> str:
        """缓存版本的先验知识过滤"""
        candidates = json.loads(candidates_hash)
        result = self._apply_prior_knowledge_logic(candidates, sentence_text)
        return json.dumps(result)
    
    def _apply_prior_knowledge(self, candidates: List[Dict[str, Any]], 
                              sentence_text: str) -> List[Dict[str, Any]]:
        """
        基于先验知识过滤候选项（带缓存）
        
        Args:
            candidates: 候选列表
            sentence_text: 句子文本
        
        Returns:
            过滤后的候选列表
        """
        # 如果候选数量很少，直接处理不缓存
        if len(candidates) <= 2:
            return self._apply_prior_knowledge_logic(candidates, sentence_text)
        
        # 转换为可缓存格式
        candidates_hash = json.dumps(candidates, sort_keys=True, ensure_ascii=False)
        result_str = self._apply_prior_knowledge_cached(candidates_hash, sentence_text)
        return json.loads(result_str)
    
    def _apply_prior_knowledge_logic(self, candidates: List[Dict[str, Any]], 
                                    sentence_text: str) -> List[Dict[str, Any]]:
        """
        先验知识过滤的实际逻辑
        
        Args:
            candidates: 候选列表
            sentence_text: 句子文本
        
        Returns:
            过滤后的候选列表
        """
        # 脊柱相关
        if re.search(self.spine_words, sentence_text):
            # 过滤出脊柱相关候选，同时排除女性附件
            filtered = [c for c in candidates 
                       if ("脊柱" in c['partlist'][0] or 
                           "椎" in "".join(c['partlist']) or
                           "脊" in "".join(c['partlist'])) and
                          "女性附件" not in c['partlist']]
            if filtered:
                return filtered
        
        # 皮肤软组织相关
        if "皮肤" in sentence_text or "软组织" in sentence_text:
            filtered = [c for c in candidates 
                       if "皮肤软组织" in "".join(c['partlist'])]
            if filtered:
                return filtered
        
        # 四肢相关
        if "上肢" in sentence_text or "下肢" in sentence_text:
            filtered = [c for c in candidates 
                       if len(c['partlist']) > 0 and 
                       ("上肢" in c['partlist'][0] or "下肢" in c['partlist'][0])]
            if filtered:
                return filtered
        
        return candidates
    
    def _match_exam_part(self, candidates: List[Dict[str, Any]], 
                        add_info: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """
        根据检查部位信息过滤
        
        Args:
            candidates: 候选列表
            add_info: 检查部位axis信息
        
        Returns:
            匹配的候选列表
        """
        matched = []
        for candidate in candidates:
            for exam_axis in add_info:
                if self._interval_cross(candidate['axis'], exam_axis):
                    matched.append(candidate)
                    break
        
        # 特殊处理：如果有匹配且数量减少，进一步过滤
        if matched and len(matched) < len(candidates):
            # 排除上肢、下肢等明显不相关的部位
            non_limb = [m for m in matched 
                       if len(m['partlist']) > 0 and
                       "上肢" not in m['partlist'][0] and 
                       "下肢" not in m['partlist'][0]]
            if non_limb:
                return non_limb
        
        return matched if matched else candidates
    
    def _match_context_partlist(self, candidates: List[Dict[str, Any]],
                               context_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于上下文实体的partlist进行匹配
        
        Args:
            candidates: 候选列表
            context_entities: 上下文实体列表
        
        Returns:
            最佳匹配的候选列表
        """
        best_matches = []
        max_score = 0
        
        for candidate in candidates:
            for context in context_entities:
                # 获取context的partlist（可能是嵌套的）
                context_partlist = self._get_first_partlist(context)
                if not context_partlist:
                    continue
                
                # 计算共同元素数量
                common_count = len(set(candidate['partlist']) & set(context_partlist))
                
                if common_count > max_score:
                    max_score = common_count
                    best_matches = [candidate]
                elif common_count == max_score and common_count > 0:
                    if candidate not in best_matches:
                        best_matches.append(candidate)
        
        return best_matches if best_matches else candidates
    
    def _match_context_axis(self, candidates: List[Dict[str, Any]],
                           context_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于上下文实体的坐标进行匹配
        
        Args:
            candidates: 候选列表
            context_entities: 上下文实体列表
        
        Returns:
            匹配的候选列表
        """
        matched = []
        for candidate in candidates:
            for context in context_entities:
                context_axis = self._get_first_axis(context)
                if context_axis and self._interval_cross(candidate['axis'], context_axis):
                    matched.append(candidate)
                    break
        
        return matched if matched else candidates
    
    def _match_extended_axis(self, candidates: List[Dict[str, Any]],
                            context_entities: List[Dict[str, Any]],
                            add_info: Optional[List[Tuple[float, float]]]) -> List[Dict[str, Any]]:
        """
        使用扩展坐标范围进行匹配
        
        Args:
            candidates: 候选列表
            context_entities: 上下文实体列表
            add_info: 检查部位信息
        
        Returns:
            匹配的候选列表
        """
        matched = []
        
        # 先尝试与上下文匹配
        for candidate in candidates:
            for context in context_entities:
                context_axis = self._get_first_axis(context)
                if context_axis and self._interval_cross(candidate['axis'], context_axis, extend=True):
                    matched.append(candidate)
                    break
        
        # 如果没有匹配，尝试与检查部位匹配
        if not matched and add_info:
            for candidate in candidates:
                for exam_axis in add_info:
                    if self._interval_cross(candidate['axis'], exam_axis, extend=True):
                        matched.append(candidate)
                        break
        
        return matched if matched else candidates
    
    def _apply_fallback_strategy(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        兜底策略：基于规则选择最可能的候选，无法确定时全部保留
        
        Args:
            candidates: 候选列表
        
        Returns:
            选中的候选列表（可能多个）
        """
        if not candidates:
            return []
        
        # 优先选择second_root中的部位
        priority_candidates = [c for c in candidates 
                              if re.search(self.second_root, c.get('position', ''))]
        if len(priority_candidates) == 1:
            return priority_candidates
        elif priority_candidates:
            candidates = priority_candidates
        
        # 选择层级较深的（更具体的）
        if len(candidates) > 1:
            max_depth = max(len(c['partlist']) for c in candidates)
            deeper_candidates = [c for c in candidates if len(c['partlist']) == max_depth]
            
            if len(deeper_candidates) == 1:
                return deeper_candidates
            
            # 如果还有多个候选，全部保留
            return deeper_candidates if deeper_candidates else candidates
        
        # 返回所有候选
        return candidates
    
    @lru_cache(maxsize=1024)
    def _interval_cross_cached(self, axis_a_str: str, axis_b_str: str, extend: bool = False) -> bool:
        """缓存版本的坐标交叉判断"""
        axis_a = json.loads(axis_a_str)
        axis_b = json.loads(axis_b_str)
        
        if not axis_a or not axis_b:
            return False
        
        if len(axis_a) < 2 or len(axis_b) < 2:
            return False
        
        try:
            if extend:
                # 扩展匹配：坐标取模100进行比较
                a_start = axis_a[0] % 100 if axis_a[0] < 500 else axis_a[0]
                a_end = axis_a[1] % 100 if axis_a[1] < 500 else axis_a[1]
                b_start = axis_b[0] % 100 if axis_b[0] < 500 else axis_b[0]
                b_end = axis_b[1] % 100 if axis_b[1] < 500 else axis_b[1]
            else:
                a_start = axis_a[0]
                a_end = axis_a[1]
                b_start = axis_b[0]
                b_end = axis_b[1]
            
            # 坐标差距过大直接返回False
            if abs(a_start - b_start) > 100:
                return False
            
            # 判断区间是否相交
            if a_start < b_end:
                return a_end > b_start
            else:
                return b_end > a_start
        except (TypeError, IndexError):
            return False
    
    def _interval_cross(self, axis_a: Tuple[float, float], 
                       axis_b: Tuple[float, float],
                       extend: bool = False) -> bool:
        """
        判断两个坐标区间是否相交（带缓存）
        
        Args:
            axis_a: 坐标区间A
            axis_b: 坐标区间B
            extend: 是否使用扩展匹配
        
        Returns:
            是否相交
        """
        # 转换为可缓存的格式
        axis_a_str = json.dumps(axis_a)
        axis_b_str = json.dumps(axis_b)
        return self._interval_cross_cached(axis_a_str, axis_b_str, extend)
    
    def _get_first_partlist(self, entity: Dict[str, Any]) -> List[str]:
        """
        获取实体的第一个partlist
        
        Args:
            entity: 实体字典
        
        Returns:
            partlist列表
        """
        partlist = entity.get('partlist', [])
        if not partlist:
            return []
        
        # 如果是嵌套列表，取第一个
        if isinstance(partlist[0], list):
            return partlist[0] if partlist else []
        else:
            # 如果不是嵌套的，直接返回
            return partlist
    
    def _get_first_axis(self, entity: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        获取实体的第一个axis
        
        Args:
            entity: 实体字典
        
        Returns:
            axis元组
        """
        axis = entity.get('axis', [])
        if not axis:
            return None
        
        # 如果是嵌套列表，取第一个
        if isinstance(axis[0], list):
            return tuple(axis[0]) if axis and len(axis[0]) >= 2 else None
        else:
            # 如果不是嵌套的，直接返回
            return tuple(axis) if len(axis) >= 2 else None
    
    def get_cache_info(self):
        """获取缓存统计信息"""
        return {
            'interval_cross': self._interval_cross_cached.cache_info(),
            'prior_knowledge': self._apply_prior_knowledge_cached.cache_info(),
            'single_ambiguity': self._resolve_single_ambiguity_cached.cache_info()
        }
    
    def clear_cache(self):
        """清理所有缓存"""
        self._interval_cross_cached.cache_clear()
        self._apply_prior_knowledge_cached.cache_clear()
        self._resolve_single_ambiguity_cached.cache_clear()


# 便捷函数
def disambiguate_entities(entities_list: List[Dict[str, Any]], 
                         add_info: Optional[List[Tuple[float, float]]] = None) -> List[Dict[str, Any]]:
    """
    对实体列表进行歧义消解的便捷函数
    
    Args:
        entities_list: 实体列表
        add_info: 检查部位信息
    
    Returns:
        消歧后的实体列表
    """
    resolver = AmbiguityResolver()
    return resolver.resolve_entities(entities_list, add_info)


# 测试函数
if __name__ == "__main__":
    import json
    
    # 模拟输入数据（来自output.json的格式）
    test_entities = [
        {
            "keyword": "支气管",
            "axis": [
                [105.0, 106.0],
                [105.0, 106.0],
                [105.0, 106.0]
            ],
            "partlist": [
                ["胸部", "肺", "肺上叶", "舌段", "支气管"],
                ["胸部", "肺", "肺上叶", "尖段", "支气管"],
                ["胸部", "肺", "肺中叶", "外侧段", "支气管"]
            ],
            "position": "支气管",
            "short_sentence": "支气管通畅",
            "sentence_index": 1
        },
        {
            "keyword": "肺",
            "axis": [[105.0, 106.0]],
            "partlist": [["胸部", "肺"]],
            "position": "肺",
            "short_sentence": "双肺未见异常密度",
            "sentence_index": 1
        }
    ]
    
    # 模拟检查部位信息
    add_info = [(105.0, 106.0)]  # 胸部检查
    
    # 执行消歧
    resolver = AmbiguityResolver()
    result = resolver.resolve_entities(test_entities, add_info)
    
    # 打印结果
    print("消歧前：")
    print(f"支气管候选数：{len(test_entities[0]['partlist'])}")
    
    print("\n消歧后：")
    print(f"支气管候选数：{len(result[0]['partlist'])}")
    print(json.dumps(result[0], indent=2, ensure_ascii=False))
    
    # 打印缓存信息
    print("\n缓存统计：")
    print(resolver.get_cache_info())