"""
自动同步 .env.example 到 .env 的工具

功能：
- 将 .env.example 中的新配置项添加到 .env
- 保留 .env 中已有的值
- 保留 .env.example 中的注释
- 不删除 .env 中多余的配置项
"""
import re
from pathlib import Path


def parse_env_with_comments(file_path: Path) -> list:
    """
    解析 env 文件，保留注释和结构
    
    Returns:
        list of dict: [{"type": "comment/blank/config", "line": "...", "key": "...", "value": "..."}]
    """
    if not file_path.exists():
        return []
    
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.rstrip('\n\r')
            
            # 空行
            if not stripped.strip():
                result.append({"type": "blank", "line": stripped})
                continue
            
            # 注释行
            if stripped.strip().startswith('#'):
                result.append({"type": "comment", "line": stripped})
                continue
            
            # 配置行 KEY=VALUE
            if '=' in stripped:
                key, _, value = stripped.partition('=')
                key = key.strip()
                result.append({
                    "type": "config",
                    "line": stripped,
                    "key": key,
                    "value": value
                })
            else:
                result.append({"type": "unknown", "line": stripped})
    
    return result


def get_config_keys(parsed: list) -> set:
    """从解析结果中提取所有配置项的 key"""
    return {item["key"] for item in parsed if item["type"] == "config"}


def sync_env_files(project_root: Path = None):
    """
    同步 .env.example 到 .env
    
    Args:
        project_root: 项目根目录，默认为当前文件的上上级目录
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    env_example_path = project_root / '.env.example'
    env_path = project_root / '.env'
    
    if not env_example_path.exists():
        print("[env_sync] .env.example 不存在，跳过同步")
        return
    
    # 解析两个文件
    example_parsed = parse_env_with_comments(env_example_path)
    env_parsed = parse_env_with_comments(env_path)
    
    # 获取配置项 key
    example_keys = get_config_keys(example_parsed)
    env_keys = get_config_keys(env_parsed)
    
    # 找出新配置项
    new_keys = example_keys - env_keys
    
    if not new_keys:
        print("[env_sync] .env 已是最新，无需同步")
        return
    
    print(f"[env_sync] 发现 {len(new_keys)} 个新配置项，正在同步...")
    
    # 收集新配置项（包含其前面的注释）
    additions = []
    for i, item in enumerate(example_parsed):
        if item["type"] == "config" and item["key"] in new_keys:
            # 向前查找相关的注释行
            comments = []
            for j in range(i - 1, -1, -1):
                prev = example_parsed[j]
                if prev["type"] in ("comment", "blank"):
                    if prev["type"] == "comment":
                        comments.insert(0, prev["line"])
                else:
                    break
            
            # 添加注释和配置项
            if comments:
                for comment in comments:
                    additions.append(comment)
            additions.append(item["line"])
            print(f"  + {item['key']}={item['value']}")
    
    # 追加到 .env 文件
    with open(env_path, 'a', encoding='utf-8') as f:
        # 确保前面有空行
        if env_parsed and env_parsed[-1].get("type") != "blank":
            f.write('\n')
        f.write('\n')
        for addition in additions:
            f.write(addition + '\n')
    
    print(f"[env_sync] 同步完成，已添加 {len(new_keys)} 个配置项")


if __name__ == '__main__':
    sync_env_files()