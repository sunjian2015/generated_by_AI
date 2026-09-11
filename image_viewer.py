#!/usr/bin/env python3
"""
局域网图片服务器
在浏览器中以画廊形式浏览指定文件夹内的图片。

用法:
    python image_server.py [图片文件夹路径] [--port 端口] [--host 主机]

示例:
    python image_server.py D:/Photos
    python image_server.py D:/Photos --port 8080
"""

from __future__ import annotations

import argparse
import html
import socket
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 支持的图片扩展名
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".webp", ".svg", ".ico", ".tiff", ".tif",
}

# 图片文件夹（运行时设置）
IMAGE_DIR: Path = Path(".")


def get_lan_ip() -> str:
    """获取本机局域网 IP 地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 不会真的发送数据，只是用来确定出口网卡
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def list_images(directory: Path) -> list[Path]:
    """列出目录下的所有图片（按名称排序）"""
    images = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda p: p.name.lower())


def list_subdirs(directory: Path) -> list[Path]:
    """列出子目录（按名称排序）"""
    subdirs = [p for p in directory.iterdir() if p.is_dir()]
    return sorted(subdirs, key=lambda p: p.name.lower())


CONTENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    ".svg": "image/svg+xml", ".ico": "image/x-icon",
    ".tiff": "image/tiff", ".tif": "image/tiff",
}


class ImageHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 简洁日志
        print(f"[{self.address_string()}] {format % args}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        rel_path = urllib.parse.unquote(parsed.path.lstrip("/"))

        # 解析目标路径，并做安全检查（防止目录穿越）
        target = (IMAGE_DIR / rel_path).resolve()
        try:
            target.relative_to(IMAGE_DIR.resolve())
        except ValueError:
            self.send_error(403, "Forbidden: 路径越界")
            return

        if not target.exists():
            self.send_error(404, "Not Found")
            return

        if target.is_dir():
            self.serve_gallery(target, rel_path)
        else:
            self.serve_file(target)

    def serve_file(self, filepath: Path):
        """发送单个图片文件"""
        ctype = CONTENT_TYPES.get(filepath.suffix.lower(), "application/octet-stream")
        try:
            data = filepath.read_bytes()
        except OSError:
            self.send_error(500, "读取文件失败")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def serve_gallery(self, directory: Path, rel_path: str):
        """生成并发送画廊 HTML 页面"""
        images = list_images(directory)
        subdirs = list_subdirs(directory)

        rel_path = rel_path.rstrip("/")
        title = rel_path if rel_path else "根目录"

        def url_for(p: Path) -> str:
            rp = p.relative_to(IMAGE_DIR.resolve())
            return "/" + urllib.parse.quote(rp.as_posix())

        # 面包屑导航
        crumbs = ['<a href="/">🏠 根目录</a>']
        if rel_path:
            parts = rel_path.split("/")
            acc = ""
            for part in parts:
                acc = f"{acc}/{part}" if acc else part
                crumbs.append(
                    f'<a href="/{urllib.parse.quote(acc)}">{html.escape(part)}</a>'
                )
        breadcrumb = ' <span class="sep">/</span> '.join(crumbs)

        # 子文件夹
        folder_items = "".join(
            f'<a class="folder" href="{url_for(d)}">'
            f'<div class="folder-icon">📁</div>'
            f'<div class="name">{html.escape(d.name)}</div></a>'
            for d in subdirs
        )

        # 图片
        image_items = "".join(
            f'<a class="card" href="{url_for(img)}" target="_blank" '
            f'data-index="{i}" data-src="{url_for(img)}" '
            f'data-name="{html.escape(img.name, quote=True)}">'
            f'<img loading="lazy" src="{url_for(img)}" alt="{html.escape(img.name)}">'
            f'<div class="name">{html.escape(img.name)}</div></a>'
            for i, img in enumerate(images)
        )

        page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>图片画廊 - {html.escape(title)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         background: #1a1a1a; color: #e0e0e0; padding: 16px; }}
  header {{ margin-bottom: 20px; }}
  h1 {{ font-size: 20px; margin-bottom: 8px; }}
  .breadcrumb {{ font-size: 14px; color: #999; margin-bottom: 8px; }}
  .breadcrumb a {{ color: #4da6ff; text-decoration: none; }}
  .breadcrumb a:hover {{ text-decoration: underline; }}
  .breadcrumb .sep {{ color: #555; }}
  .count {{ font-size: 13px; color: #777; }}
  .grid {{ display: grid;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 12px; }}
  .folder, .card {{ background: #2a2a2a; border-radius: 8px; overflow: hidden;
                    text-decoration: none; color: #e0e0e0;
                    transition: transform .15s, box-shadow .15s; display: block; }}
  .folder:hover, .card:hover {{ transform: translateY(-3px);
                    box-shadow: 0 6px 16px rgba(0,0,0,.4); }}
  .card img {{ width: 100%; height: 160px; object-fit: cover; display: block;
               background: #333; }}
  .folder-icon {{ font-size: 64px; text-align: center; padding: 30px 0 10px; }}
  .name {{ padding: 8px 10px; font-size: 13px; white-space: nowrap;
           overflow: hidden; text-overflow: ellipsis; }}
  .empty {{ color: #777; padding: 40px; text-align: center; }}
  /* 灯箱 */
  #lightbox {{ display: none; position: fixed; inset: 0;
               background: rgba(0,0,0,.92); z-index: 999;
               align-items: center; justify-content: center; }}
  #lightbox img {{ max-width: 92%; max-height: 90%; object-fit: contain;
                   user-select: none; }}
  #lb-caption {{ position: fixed; bottom: 16px; left: 50%;
                 transform: translateX(-50%); color: #ddd; font-size: 14px;
                 background: rgba(0,0,0,.5); padding: 6px 14px; border-radius: 20px;
                 max-width: 80%; white-space: nowrap; overflow: hidden;
                 text-overflow: ellipsis; }}
  .lb-nav {{ position: fixed; top: 50%; transform: translateY(-50%);
             font-size: 48px; color: #fff; background: rgba(0,0,0,.3);
             border: none; cursor: pointer; padding: 12px 20px; border-radius: 8px;
             user-select: none; transition: background .15s; line-height: 1; }}
  .lb-nav:hover {{ background: rgba(255,255,255,.15); }}
  #lb-prev {{ left: 12px; }}
  #lb-next {{ right: 12px; }}
  #lb-close {{ position: fixed; top: 12px; right: 16px; font-size: 32px;
               color: #fff; background: none; border: none; cursor: pointer; }}
</style>
</head>
<body>
<header>
  <h1>📷 图片画廊</h1>
  <div class="breadcrumb">{breadcrumb}</div>
  <div class="count">{len(subdirs)} 个文件夹 · {len(images)} 张图片</div>
</header>
<div class="grid">
  {folder_items}
  {image_items}
</div>
{"" if (images or subdirs) else '<div class="empty">此文件夹为空</div>'}

<div id="lightbox">
  <button id="lb-close" title="关闭 (Esc)">&times;</button>
  <button class="lb-nav" id="lb-prev" title="上一张 (←)">&#8249;</button>
  <img src="" alt="">
  <button class="lb-nav" id="lb-next" title="下一张 (→)">&#8250;</button>
  <div id="lb-caption"></div>
</div>
<script>
  // 收集所有图片卡片，供灯箱切换使用
  const cards = Array.from(document.querySelectorAll('.card'));
  const lb = document.getElementById('lightbox');
  const lbImg = lb.querySelector('img');
  const lbCaption = document.getElementById('lb-caption');
  let currentIndex = -1;

  function openLightbox(index) {{
    if (index < 0 || index >= cards.length) return;
    currentIndex = index;
    const card = cards[index];
    lbImg.src = card.dataset.src;
    lbCaption.textContent =
      card.dataset.name + '  (' + (index + 1) + ' / ' + cards.length + ')';
    lb.style.display = 'flex';
  }}

  function closeLightbox() {{
    lb.style.display = 'none';
    lbImg.src = '';
    currentIndex = -1;
  }}

  function showPrev() {{ openLightbox((currentIndex - 1 + cards.length) % cards.length); }}
  function showNext() {{ openLightbox((currentIndex + 1) % cards.length); }}

  cards.forEach(card => {{
    card.addEventListener('click', e => {{
      e.preventDefault();
      openLightbox(parseInt(card.dataset.index, 10));
    }});
  }});

  // 按钮点击（阻止冒泡，避免触发背景的关闭）
  document.getElementById('lb-close').addEventListener('click', e => {{
    e.stopPropagation(); closeLightbox();
  }});
  document.getElementById('lb-prev').addEventListener('click', e => {{
    e.stopPropagation(); showPrev();
  }});
  document.getElementById('lb-next').addEventListener('click', e => {{
    e.stopPropagation(); showNext();
  }});
  // 点击图片本身不关闭，点击背景才关闭
  lbImg.addEventListener('click', e => e.stopPropagation());
  lb.addEventListener('click', closeLightbox);

  // 键盘导航
  document.addEventListener('keydown', e => {{
    if (lb.style.display !== 'flex') return;
    if (e.key === 'Escape') closeLightbox();
    else if (e.key === 'ArrowLeft') {{ e.preventDefault(); showPrev(); }}
    else if (e.key === 'ArrowRight') {{ e.preventDefault(); showNext(); }}
  }});
</script>
</body>
</html>"""

        data = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global IMAGE_DIR

    parser = argparse.ArgumentParser(description="局域网图片画廊服务器")
    parser.add_argument("directory", nargs="?", default=".",
                        help="图片文件夹路径（默认为当前目录）")
    parser.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    parser.add_argument("--host", default="0.0.0.0",
                        help="绑定地址（默认 0.0.0.0，允许局域网访问）")
    args = parser.parse_args()

    IMAGE_DIR = Path(args.directory).resolve()
    if not IMAGE_DIR.is_dir():
        print(f"错误: 文件夹不存在: {IMAGE_DIR}")
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), ImageHandler)
    lan_ip = get_lan_ip()

    print("=" * 50)
    print("  局域网图片画廊服务器已启动")
    print("=" * 50)
    print(f"  图片目录: {IMAGE_DIR}")
    print(f"  本机访问: http://127.0.0.1:{args.port}")
    print(f"  局域网访问: http://{lan_ip}:{args.port}")
    print("=" * 50)
    print("  按 Ctrl+C 停止服务")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
        server.shutdown()
        print("已停止。")


if __name__ == "__main__":
    main()
