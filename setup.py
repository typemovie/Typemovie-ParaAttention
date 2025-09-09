#!/usr/bin/env python3

import importlib.util
import os
import subprocess
from os import path
from setuptools import find_packages, setup

if importlib.util.find_spec("setuptools_scm") is None:
    raise ImportError("setuptools-scm is not installed. Install it by `pip3 install setuptools-scm`")

# 设置环境变量以避免手动输入 - 注意包名变化后环境变量名也要相应改变
os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TYPEMOVIE_PARAATTN", "0.1")

def is_git_directory(path="."):
    return subprocess.call(["git", "-C", path, "status"], stderr=subprocess.STDOUT, stdout=open(os.devnull, "w")) == 0


def my_local_scheme(version):
    local_version = os.getenv("PARA_ATTN_BUILD_LOCAL_VERSION")
    if local_version is None:
        from setuptools_scm.version import get_local_dirty_tag
        return get_local_dirty_tag(version)
    return f"+{local_version}"


def fetch_requirements():
    """获取依赖列表，如果 requirements.txt 不存在则返回空列表"""
    requirements_file = "requirements.txt"
    if path.exists(requirements_file):
        with open(requirements_file) as f:
            reqs = f.read().strip().split("\n")
        return [req for req in reqs if req.strip() and not req.strip().startswith('#')]
    else:
        print(f"Warning: {requirements_file} not found, using minimal dependencies.")
        return []


setup(
    name="typemovie-paraattn",  # 修改包名
    use_scm_version={
        "write_to": path.join("src", "para_attn", "_version.py"),  # 源码目录保持不变
        "local_scheme": my_local_scheme,
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=fetch_requirements(),
    extras_require={
        "all": [],
        "dev": [
            "pre-commit",
            "pytest>=7.0.0,<8.0.0",
            "pytest-html",
            "expecttest",
            "hypothesis",
            "transformers",
            "diffusers<=0.34",
            "accelerate",
            "peft",
            "protobuf",
            "sentencepiece",
            "opencv-python",
            "ftfy",
        ],
    },
    license="MIT",
)