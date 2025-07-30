#!/usr/bin/env python3
"""
prepare_blog_char.py
--------------------
• 递归读取 --posts_dir 目录内的 *.md
• 去掉 YAML front-matter、代码块、Markdown 链接/标签，只保留正文纯文本
• 生成字符级 vocab，编码成整数序列
• 90/10 切分为 train/val
• 写出 train.bin / val.bin 及 meta.pkl
兼容 nanoGPT 的 char 训练脚本。
"""

import argparse, os, re, pickle, numpy as np
from pathlib import Path
from tqdm import tqdm

FRONT_MAT_RE = re.compile(r'^---[\s\S]*?---\s*', re.MULTILINE)           # YAML front-matter
CODEBLOCK_RE = re.compile(r'```[\s\S]*?```')                             # 三反引号代码块
LINK_RE      = re.compile(r'\[([^\]]+)\]\([^)]+\)')                      # [text](url) → text
INLINE_CODE  = re.compile(r'`([^`]+)`')                                  # `code` → code
HTML_TAGS    = re.compile(r'<[^>]+>')                                    # 粗暴去 HTML/标签
HEADER_RE    = re.compile(r'^#+\s*', re.MULTILINE)                       # ## heading 标记
IMG_RE       = re.compile(r'!\[.*?\]\(.*?\)')                            # 图片语法

def clean_markdown(md: str) -> str:
    md = FRONT_MAT_RE.sub('', md)
    md = CODEBLOCK_RE.sub('', md)
    md = IMG_RE.sub('', md)
    md = LINK_RE.sub(r'\1', md)
    md = INLINE_CODE.sub(r'\1', md)
    md = HTML_TAGS.sub('', md)
    md = HEADER_RE.sub('', md)
    # 多余空行压缩
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip() + '\n\n'

def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi  = { ch:i for i,ch in enumerate(chars) }
    itos  = { i:ch for ch,i in stoi.items() }
    return chars, stoi, itos

def encode(text: str, stoi):
    return [stoi[c] for c in text]

def main(args):
    posts_dir = Path(args.posts_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取并清洗所有 md
    md_files = list(posts_dir.rglob('*.md'))
    print(f'Found {len(md_files)} markdown files under {posts_dir}')
    corpus_parts = []
    for fp in tqdm(md_files, desc='reading'):
        corpus_parts.append(clean_markdown(fp.read_text(encoding='utf-8', errors='ignore')))
    corpus = '\n'.join(corpus_parts)

    # 2. 构建 vocab & 编码
    chars, stoi, itos = build_vocab(corpus)
    print(f'Vocab size: {len(chars)} unique characters')

    ids = np.array(encode(corpus, stoi), dtype=np.uint16 if len(chars) < 65536 else np.uint32)
    n   = len(ids)
    split = int(n * args.train_frac)
    train_ids, val_ids = ids[:split], ids[split:]

    # 3. 写文件
    train_ids.tofile(out_dir / 'train.bin')
    val_ids.tofile(out_dir / 'val.bin')
    meta = {
        'vocab_size': len(chars),
        'itos': itos,
        'stoi': stoi,
        'dtype': str(ids.dtype),
    }
    with open(out_dir / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    print(f'Done. Wrote:',
          f'\n  {out_dir / "train.bin"} ({train_ids.nbytes/1e6:.2f} MB)',
          f'\n  {out_dir / "val.bin"}   ({val_ids.nbytes/1e6:.2f} MB)',
          f'\n  {out_dir / "meta.pkl"}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare blog md files for nanoGPT char training")
    parser.add_argument('--posts_dir', type=str, default='dataset/posts',
                        help='directory containing markdown files')
    parser.add_argument('--out_dir',   type=str, default='dataset/blog_char',
                        help='output directory')
    parser.add_argument('--train_frac', type=float, default=0.9,
                        help='fraction of data for training (rest for val)')
    args = parser.parse_args()
    main(args)
