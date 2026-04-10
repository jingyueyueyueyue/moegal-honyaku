"""
自动同步 .env.example 到 .env 的工具

功能：
- 将 .env.example 的完整格式同步到 .env
- 保留 .env 中已有的配置值
- 新配置项使用 .env.example 的默认值
- 同步后 .env 格式与 .env.example 一致
"""
import os
import re
from pathlib import Path


def parse_env_file(file_path: Path) -> dict:
    """解析 env 文件，返回 {key: value} 字典"""
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
                result[key] = value
    
    return result


def sync_env_files(project_root: Path = None):
    """
    同步 .env.example 到 .env
    
    同步后的 .env 会保持与 .env.example 相同的格式，
    同时保留 .env 中已有的配置值。
    
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
    
    # 解析已有的配置值
    existing_values = parse_env_file(env_path)
    
    # 读取 .env.example 内容
    with open(env_example_path, 'r', encoding='utf-8') as f:
        example_content = f.read()
    
    # 统计变化
    new_keys = []
    updated_keys = []
    
    # 生成新的 .env 内容
    new_lines = []
    for line in example_content.splitlines():
        stripped = line.strip()
        
        # 注释和空行直接保留
        if not stripped or stripped.startswith('#'):
            new_lines.append(line)
            continue
        
        # 解析配置项
        if '=' in stripped:
            key, _, default_value = stripped.partition('=')
            key = key.strip()
            default_value = default_value.strip()
            
            # 如果已有配置，使用已有值
            if key in existing_values:
                existing_value = existing_values[key]
                if existing_value != default_value:
                    new_lines.append(f"{key}={existing_value}")
                    updated_keys.append(key)
                else:
                    new_lines.append(line)
            else:
                # 新配置项，使用默认值
                new_lines.append(line)
                new_keys.append(key)
        else:
            new_lines.append(line)
    
    # 写入 .env
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
        if not example_content.endswith('\n'):
            f.write('\n')
    
    # 输出同步结果
    if new_keys or updated_keys:
        print(f"[env_sync] 同步完成")
        if new_keys:
            print(f"  新增配置项: {', '.join(new_keys)}")
        if updated_keys:
            print(f"  保留自定义值: {', '.join(updated_keys)}")
    else:
        print("[env_sync] .env 已是最新，无需更新")


if __name__ == '__main__':
    sync_env_files()
