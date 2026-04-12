#!/usr/bin/env python3
"""
Translate Markdown files from `docs/en/` into `docs/vi/` using Google (deep-translator).
This script preserves YAML frontmatter, fenced code blocks, and inline code spans.

Usage:
  tests/venv/bin/python scripts/translate_to_vi.py

Note: This is a best-effort machine translation draft generator — always review results.
"""
from pathlib import Path
import re
import sys

try:
    from deep_translator import GoogleTranslator
except Exception as e:
    print('Missing dependency: deep-translator. Install with `pip install deep-translator`')
    raise

SRC = Path('docs/en')
DST = Path('docs/vi')
translator = GoogleTranslator(source='en', target='vi')

code_fence_re = re.compile(r'^```')
inline_code_re = re.compile(r'`[^`]+`')


def translate_text_with_placeholders(text: str) -> str:
    # replace inline code with placeholders
    codes = inline_code_re.findall(text)
    masked = text
    placeholders = {}
    for i, c in enumerate(codes):
        token = f"__CODE_{i}__"
        placeholders[token] = c
        masked = masked.replace(c, token)
    # translate masked text
    try:
        translated = translator.translate(masked)
    except Exception as e:
        print('Translation error:', e)
        translated = masked
    if translated is None:
        translated = masked
    # restore placeholders
    for token, c in placeholders.items():
        translated = translated.replace(token, c)
    return translated


def translate_file(src_path: Path, dst_path: Path):
    text = src_path.read_text(encoding='utf-8')
    # handle YAML frontmatter using robust regex
    front = ''
    rest = text
    if text.startswith('---'):
        m = re.match(r'(?s)^---\n.*?\n---\n', text)
        if m:
            front = m.group(0)
            rest = text[m.end():]
        else:
            # malformed frontmatter; treat whole as body
            front = ''
            rest = text
    lines = rest.splitlines(keepends=True)
    out_lines = []
    in_code = False
    for ln in lines:
        if code_fence_re.match(ln):
            in_code = not in_code
            out_lines.append(ln)
            continue
        if in_code:
            out_lines.append(ln)
            continue
        # not in code block — translate line-aware
        stripped = ln.rstrip('\n')
        if stripped.strip() == '':
            out_lines.append(ln)
            continue
        # headings
        m = re.match(r'^(\s*)(#+)(\s*)(.*)$', stripped)
        if m:
            pre, hashes, sp, txt = m.groups()
            new_txt = translate_text_with_placeholders(txt)
            out_lines.append(f"{pre}{hashes}{sp}{new_txt}\n")
            continue
        # list items
        m = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.*)$', stripped)
        if m:
            pre, marker, txt = m.groups()
            new_txt = translate_text_with_placeholders(txt)
            out_lines.append(f"{pre}{marker} {new_txt}\n")
            continue
        # blockquote
        m = re.match(r'^(\s*)(>+)\s*(.*)$', stripped)
        if m:
            pre, markers, txt = m.groups()
            new_txt = translate_text_with_placeholders(txt)
            out_lines.append(f"{pre}{markers} {new_txt}\n")
            continue
        # normal paragraph line
        new_txt = translate_text_with_placeholders(stripped)
        out_lines.append(new_txt + ('\n' if ln.endswith('\n') else ''))

    # prepend a DRAFT notice after frontmatter (or at top)
    draft_notice = ("!!! warning \"DRAFT (Machine Translation)\"\n"
                    "    This page was generated automatically by machine translation.\n"
                    "    Please review and post-edit before publishing.\n\n")
    if front:
        final = front + draft_notice + ''.join(out_lines)
    else:
        final = draft_notice + ''.join(out_lines)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(final, encoding='utf-8')
    print('Translated:', src_path, '→', dst_path)


def main():
    files = list(SRC.rglob('*.md'))
    if not files:
        print('No markdown files found in', SRC)
        return
    for f in files:
        rel = f.relative_to(SRC)
        target = DST / rel
        try:
            translate_file(f, target)
        except Exception as e:
            print('Failed:', f, e)
    print('Translation run completed. Review `docs/vi/` for drafts.')


if __name__ == '__main__':
    main()
