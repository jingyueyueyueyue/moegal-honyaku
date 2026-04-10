"""
自动同步 .env.example 到 .env 的工具

功能：
- 将 .env.example 中的新配置项添加到 .env
- 保留 .env 中已有的值
- 不删除 .env 中多余的配置项
"""
import os
import re
from pathlib import Path


def parse_env_file(file_path: Path) -> dict:
    """解析 env 文件，返回 {key: (value, line)} 字典"""
    if not file_path.exists():
        return {}
    
    result = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            # 跳过注释和空行
            if not stripped or stripped.startswith('#'):
                continue
            # 解析 KEY=VALUE
            if '=' in stripped:
                key, _, value = stripped.partition('=')
                key = key.strip()
                result[key] = (value, line.rstrip('\n\r'))
    return result


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
    example_config = parse_env_file(env_example_path)
    env_config = parse_env_file(env_path)
    
    # 找出新配置项
    new_keys = set(example_config.keys()) - set(env_config.keys())
    
    if not new_keys:
        print("[env_sync] .env 已是最新，无需同步")
        return
    
    print(f"[env_sync] 发现 {len(new_keys)} 个新配置项，正在同步...")
    
    # 读取 .env 内容
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            env_content = f.read()
    else:
        env_content = ""
    
    # 添加新配置项
    additions = []
    for key in sorted(new_keys):
        value, _ = example_config[key]
        additions.append(f"{key}={value}")
        print(f"  + {key}={value}")
    
    # 追加到 .env 文件
    with open(env_path, 'a', encoding='utf-8') as f:
        if env_content and not env_content.endswith('\n'):
            f.write('\n')
        f.write('\n# ===== 自动同步的新配置项 =====\n')
        for addition in additions:
            f.write(addition + '\n')
    
    print(f"[env_sync] 同步完成，已添加 {len(new_keys)} 个配置项")


if __name__ == '__main__':
    sync_env_files()
