#!/usr/bin/env python3
"""
局域网图片对比服务器
简洁的图片对比工具，支持在浏览器中选择两个文件夹进行对比。

用法:
    python image_compare.py [根目录] [--port 端口]

示例:
    python image_compare.py D:/Projects
    python image_compare.py .
"""

from __future__ import annotations

import argparse
import html
import json
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

# 根目录（运行时设置）
ROOT_DIR: Path = Path(".")


def get_lan_ip() -> str:
    """获取本机局域网 IP 地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def list_images(directory: Path) -> list[Path]:
    """列出目录下的所有图片（按名称排序）"""
    try:
        images = [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return sorted(images, key=lambda p: p.name.lower())
    except (PermissionError, OSError):
        return []


def get_all_subdirs(root: Path) -> list[tuple[str, str]]:
    """递归获取所有子文件夹，返回 (相对路径, 显示名称)"""
    result = []
    
    def scan(directory: Path, prefix: str = ""):
        try:
            for item in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
                if item.is_dir():
                    rel_path = str(item.relative_to(root))
                    display = prefix + item.name
                    result.append((rel_path, display))
                    scan(item, prefix + "  ")
        except (PermissionError, OSError):
            pass
    
    scan(root)
    return result


def find_matching_pairs(dir_a: Path, dir_b: Path) -> list[tuple[str, str, str]]:
    """找出两个文件夹中文件名相同的图片对"""
    images_a = {img.name: img for img in list_images(dir_a)}
    images_b = {img.name: img for img in list_images(dir_b)}
    
    common_names = sorted(set(images_a.keys()) & set(images_b.keys()))
    
    return [(name, str(images_a[name].relative_to(ROOT_DIR)), 
             str(images_b[name].relative_to(ROOT_DIR))) 
            for name in common_names]


CONTENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    ".svg": "image/svg+xml", ".ico": "image/x-icon",
    ".tiff": "image/tiff", ".tif": "image/tiff",
}


class CompareHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path.lstrip("/"))

        if not path or path == "index.html":
            self.serve_main_page()
        elif path == "api/folders":
            self.serve_folders_api()
        elif path.startswith("api/compare"):
            query = urllib.parse.parse_qs(parsed.query)
            dir_a = query.get('a', [''])[0]
            dir_b = query.get('b', [''])[0]
            self.serve_compare_api(dir_a, dir_b)
        elif path.startswith("image/"):
            img_path = path[6:]
            self.serve_image(ROOT_DIR / img_path)
        else:
            self.send_error(404, "Not Found")

    def serve_image(self, filepath: Path):
        """发送图片文件"""
        try:
            filepath = filepath.resolve()
            filepath.relative_to(ROOT_DIR.resolve())
        except (ValueError, OSError):
            self.send_error(403, "Forbidden")
            return
        
        if not filepath.exists() or not filepath.is_file():
            self.send_error(404, "Not Found")
            return
        
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

    def serve_folders_api(self):
        """返回所有子文件夹列表"""
        folders = get_all_subdirs(ROOT_DIR)
        data = json.dumps(folders, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_compare_api(self, dir_a_rel: str, dir_b_rel: str):
        """返回对比数据"""
        try:
            dir_a = (ROOT_DIR / dir_a_rel).resolve()
            dir_b = (ROOT_DIR / dir_b_rel).resolve()
            
            dir_a.relative_to(ROOT_DIR.resolve())
            dir_b.relative_to(ROOT_DIR.resolve())
            
            if not dir_a.is_dir() or not dir_b.is_dir():
                raise ValueError("文件夹不存在")
            
            pairs = find_matching_pairs(dir_a, dir_b)
            
            images_a = set(img.name for img in list_images(dir_a))
            images_b = set(img.name for img in list_images(dir_b))
            
            result = {
                'pairs': [
                    {
                        'name': name,
                        'urlA': f"/image/{urllib.parse.quote(path_a.replace(chr(92), '/'))}",
                        'urlB': f"/image/{urllib.parse.quote(path_b.replace(chr(92), '/'))}"
                    }
                    for name, path_a, path_b in pairs
                ],
                'onlyA': len(images_a - images_b),
                'onlyB': len(images_b - images_a),
                'dirA': dir_a.name,
                'dirB': dir_b.name,
            }
            
            data = json.dumps(result, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            
        except Exception as e:
            error = {'error': str(e)}
            data = json.dumps(error).encode('utf-8')
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def serve_main_page(self):
        """主页面"""
        page = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>图片对比工具</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
         min-height: 100vh; display: flex; align-items: center; justify-content: center;
         padding: 20px; }
  
  /* 选择界面 */
  #select-screen { background: #fff; border-radius: 16px; padding: 40px;
                   max-width: 600px; width: 100%; box-shadow: 0 20px 60px rgba(0,0,0,.3); }
  h1 { font-size: 28px; margin-bottom: 10px; color: #333; text-align: center; }
  .subtitle { text-align: center; color: #999; font-size: 14px; margin-bottom: 30px; }
  
  .step { margin-bottom: 24px; }
  .step-title { font-size: 16px; color: #555; margin-bottom: 8px; font-weight: 500; }
  .step-title .num { display: inline-block; width: 24px; height: 24px;
                     background: #667eea; color: #fff; border-radius: 50%;
                     text-align: center; line-height: 24px; font-size: 14px;
                     margin-right: 8px; }
  
  select { width: 100%; padding: 12px 16px; font-size: 15px; border: 2px solid #e0e0e0;
           border-radius: 8px; background: #fff; cursor: pointer;
           transition: border-color .2s; }
  select:focus { outline: none; border-color: #667eea; }
  select:disabled { background: #f5f5f5; cursor: not-allowed; }
  
  #start-btn { width: 100%; padding: 16px; font-size: 16px; font-weight: 600;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               color: #fff; border: none; border-radius: 8px; cursor: pointer;
               margin-top: 30px; transition: transform .2s, box-shadow .2s; }
  #start-btn:hover:not(:disabled) { transform: translateY(-2px);
                                    box-shadow: 0 8px 20px rgba(102,126,234,.4); }
  #start-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  
  .info { background: #f0f4ff; padding: 12px 16px; border-radius: 8px;
          font-size: 13px; color: #667eea; margin-top: 20px; line-height: 1.6; }
  .info strong { color: #764ba2; }
  
  /* 对比界面 */
  #compare-screen { display: none; position: fixed; inset: 0; background: #1a1a1a;
                    flex-direction: column; }
  #compare-header { display: flex; justify-content: space-between; align-items: center;
                    padding: 16px 20px; background: #2a2a2a; }
  #compare-title { font-size: 16px; color: #fff; flex: 1; text-align: center;
                   white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                   margin: 0 20px; }
  #back-btn { padding: 8px 20px; background: #555; color: #fff; border: none;
              border-radius: 6px; cursor: pointer; font-size: 14px; }
  #back-btn:hover { background: #666; }
  
  #compare-container { flex: 1; display: flex; align-items: center;
                       justify-content: center; gap: 12px; padding: 20px;
                       position: relative; }
  .panel { flex: 1; display: flex; flex-direction: column; align-items: center;
           max-width: 50%; height: 100%; }
  .panel-label { font-size: 15px; color: #aaa; margin-bottom: 12px;
                 padding: 6px 16px; background: rgba(0,0,0,.4); border-radius: 20px; }
  .img-box { flex: 1; display: flex; align-items: center; justify-content: center;
             width: 100%; overflow: hidden; position: relative; }
  .img-box.zoomed { cursor: grab; }
  .img-box.dragging { cursor: grabbing; }
  .img-box img { max-width: 100%; max-height: 100%; object-fit: contain;
                 user-select: none; pointer-events: none;
                 position: relative; }
  
  .nav-btn { position: fixed; top: 50%; transform: translateY(-50%);
             font-size: 48px; color: #fff; background: rgba(0,0,0,.6);
             border: none; cursor: pointer; padding: 16px 24px; border-radius: 12px;
             transition: background .2s; z-index: 10; }
  .nav-btn:hover { background: rgba(255,255,255,.2); }
  #prev-btn { left: 20px; }
  #next-btn { right: 20px; }
  
  #counter { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
             background: rgba(0,0,0,.8); padding: 10px 24px; border-radius: 25px;
             font-size: 15px; color: #fff; z-index: 10; font-weight: 500; }
  
  /* 缩放控制 */
  #zoom-controls { position: fixed; bottom: 20px; right: 20px; z-index: 10;
                   display: none; flex-direction: column; gap: 8px; }
  .zoom-btn { width: 44px; height: 44px; background: rgba(0,0,0,.8);
              color: #fff; border: none; border-radius: 8px; cursor: pointer;
              font-size: 20px; transition: background .2s; }
  .zoom-btn:hover { background: rgba(255,255,255,.2); }
  #zoom-level { background: rgba(0,0,0,.8); color: #fff; padding: 6px 12px;
                border-radius: 8px; font-size: 13px; text-align: center; }
  
  /* 循环开关 */
  #loop-control { position: fixed; bottom: 20px; left: 20px; z-index: 10;
                  display: none; background: rgba(0,0,0,.8); padding: 10px 16px;
                  border-radius: 8px; color: #fff; font-size: 14px;
                  align-items: center; gap: 8px; cursor: pointer;
                  transition: background .2s; user-select: none; }
  #loop-control:hover { background: rgba(255,255,255,.15); }
  #loop-toggle { width: 40px; height: 22px; background: #555; border-radius: 11px;
                 position: relative; transition: background .2s; cursor: pointer; }
  #loop-toggle.active { background: #667eea; }
  #loop-toggle::after { content: ''; position: absolute; width: 18px; height: 18px;
                        background: #fff; border-radius: 50%; top: 2px; left: 2px;
                        transition: left .2s; }
  #loop-toggle.active::after { left: 20px; }
  
  /* 提示消息 */
  #toast { position: fixed; top: 80px; left: 50%; transform: translateX(-50%);
           background: rgba(0,0,0,.9); color: #fff; padding: 12px 24px;
           border-radius: 8px; font-size: 15px; z-index: 1001;
           display: none; animation: fadeInOut 2s ease-in-out; }
  @keyframes fadeInOut {
    0%, 100% { opacity: 0; }
    10%, 90% { opacity: 1; }
  }
  
  #loading { position: fixed; inset: 0; background: rgba(0,0,0,.9);
             display: none; align-items: center; justify-content: center;
             z-index: 1000; }
  .spinner { border: 4px solid rgba(255,255,255,.1);
             border-top-color: #667eea; border-radius: 50%;
             width: 50px; height: 50px; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  
  .error { background: #ffebee; color: #c62828; padding: 16px; border-radius: 8px;
           margin-top: 16px; font-size: 14px; text-align: center; display: none; }
</style>
</head>
<body>

<!-- 选择界面 -->
<div id="select-screen">
  <h1>🔍 图片对比工具</h1>
  <div class="subtitle">选择两个文件夹，对比其中的同名图片</div>
  
  <div class="step">
    <div class="step-title"><span class="num">1</span>选择第一个文件夹</div>
    <select id="folder-a">
      <option value="">-- 请选择文件夹 A --</option>
    </select>
  </div>
  
  <div class="step">
    <div class="step-title"><span class="num">2</span>选择第二个文件夹</div>
    <select id="folder-b">
      <option value="">-- 请选择文件夹 B --</option>
    </select>
  </div>
  
  <button id="start-btn" disabled>开始对比</button>
  
  <div class="info">
    <strong>💡 提示：</strong>对比时会找出两个文件夹中文件名相同的图片，左右并排显示。<br>
    使用键盘 <strong>← →</strong> 方向键可以快速切换图片。
  </div>
  
  <div class="error" id="error-msg"></div>
</div>

<!-- 对比界面 -->
<div id="compare-screen">
  <div id="compare-header">
    <button id="back-btn">← 返回</button>
    <div id="compare-title"></div>
    <div style="width: 80px;"></div>
  </div>
  <div id="compare-container">
    <button class="nav-btn" id="prev-btn">‹</button>
    <div class="panel">
      <div class="panel-label" id="label-a">文件夹 A</div>
      <div class="img-box"><img id="img-a" src="" alt=""></div>
    </div>
    <div class="panel">
      <div class="panel-label" id="label-b">文件夹 B</div>
      <div class="img-box"><img id="img-b" src="" alt=""></div>
    </div>
    <button class="nav-btn" id="next-btn">›</button>
  </div>
  <div id="counter"></div>
  <div id="loop-control">
    <span>循环播放</span>
    <div id="loop-toggle" class="active"></div>
  </div>
  <div id="zoom-controls">
    <button class="zoom-btn" id="zoom-in" title="放大 (+)">+</button>
    <div id="zoom-level">100%</div>
    <button class="zoom-btn" id="zoom-out" title="缩小 (-)">−</button>
    <button class="zoom-btn" id="zoom-reset" title="重置 (0)">⊙</button>
  </div>
</div>

<div id="toast"></div>
<div id="loading"><div class="spinner"></div></div>

<script>
  let folders = [];
  let pairs = [];
  let currentIndex = -1;
  const imageCache = new Map();
  let zoomLevel = 1.0;
  let loopEnabled = true;
  
  const selectScreen = document.getElementById('select-screen');
  const compareScreen = document.getElementById('compare-screen');
  const loading = document.getElementById('loading');
  const folderA = document.getElementById('folder-a');
  const folderB = document.getElementById('folder-b');
  const startBtn = document.getElementById('start-btn');
  const errorMsg = document.getElementById('error-msg');
  const imgA = document.getElementById('img-a');
  const imgB = document.getElementById('img-b');
  const compareTitle = document.getElementById('compare-title');
  const counter = document.getElementById('counter');
  const labelA = document.getElementById('label-a');
  const labelB = document.getElementById('label-b');
  const zoomControls = document.getElementById('zoom-controls');
  const zoomLevelDisplay = document.getElementById('zoom-level');
  const loopControl = document.getElementById('loop-control');
  const loopToggle = document.getElementById('loop-toggle');
  const toast = document.getElementById('toast');
  
  // 拖拽状态
  let pan = { x: 0, y: 0 };
  let dragState = null;
  
  function preloadImage(url) {
    if (!imageCache.has(url)) {
      const img = new Image();
      img.src = url;
      imageCache.set(url, img);
    }
    return imageCache.get(url);
  }
  
  // 提示消息
  function showToast(msg) {
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.display = 'none';
    }, 2000);
  }
  
  // 循环开关
  loopControl.addEventListener('click', () => {
    loopEnabled = !loopEnabled;
    loopToggle.classList.toggle('active', loopEnabled);
    showToast(loopEnabled ? '已开启循环播放' : '已关闭循环播放');
  });
  
  // 缩放功能
  function setZoom(level) {
    zoomLevel = Math.max(0.5, Math.min(5, level));
    updateTransform();
    zoomLevelDisplay.textContent = Math.round(zoomLevel * 100) + '%';
    
    // 更新光标样式
    const boxes = document.querySelectorAll('.img-box');
    if (zoomLevel > 1) {
      boxes.forEach(box => box.classList.add('zoomed'));
    } else {
      boxes.forEach(box => box.classList.remove('zoomed'));
      // 缩放为1时重置平移
      pan = { x: 0, y: 0 };
      updateTransform();
    }
  }
  
  function updateTransform() {
    imgA.style.transform = `translate(${pan.x}px, ${pan.y}px) scale(${zoomLevel})`;
    imgB.style.transform = `translate(${pan.x}px, ${pan.y}px) scale(${zoomLevel})`;
  }
  
  function zoomIn() { setZoom(zoomLevel * 1.2); }
  function zoomOut() { setZoom(zoomLevel / 1.2); }
  function zoomReset() { 
    pan = { x: 0, y: 0 };
    setZoom(1.0); 
  }
  
  // 拖拽功能 - 两张图片同步拖动
  document.getElementById('compare-container').addEventListener('mousedown', e => {
    if (zoomLevel <= 1) return;
    
    const imgBox = e.target.closest('.img-box');
    if (!imgBox) return;
    
    e.preventDefault();
    
    dragState = {
      imgBox,
      startX: e.clientX - pan.x,
      startY: e.clientY - pan.y
    };
    imgBox.classList.add('dragging');
    // 同时给另一个框也加上拖动状态
    document.querySelectorAll('.img-box').forEach(box => box.classList.add('dragging'));
  });
  
  document.addEventListener('mousemove', e => {
    if (!dragState) return;
    e.preventDefault();
    pan.x = e.clientX - dragState.startX;
    pan.y = e.clientY - dragState.startY;
    updateTransform();
  });
  
  document.addEventListener('mouseup', () => {
    if (dragState) {
      document.querySelectorAll('.img-box').forEach(box => box.classList.remove('dragging'));
      dragState = null;
    }
  });
  
  document.getElementById('zoom-in').addEventListener('click', zoomIn);
  document.getElementById('zoom-out').addEventListener('click', zoomOut);
  document.getElementById('zoom-reset').addEventListener('click', zoomReset);
  
  // 鼠标滚轮缩放
  document.getElementById('compare-container').addEventListener('wheel', e => {
    if (compareScreen.style.display !== 'flex') return;
    e.preventDefault();
    if (e.deltaY < 0) {
      zoomIn();
    } else {
      zoomOut();
    }
  }, { passive: false });
  
  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.style.display = 'block';
    setTimeout(() => errorMsg.style.display = 'none', 5000);
  }
  
  // 加载文件夹列表
  async function loadFolders() {
    loading.style.display = 'flex';
    try {
      const res = await fetch('/api/folders');
      folders = await res.json();
      
      updateFolderOptions();
      
      if (folders.length === 0) {
        showError('当前目录下没有子文件夹');
        startBtn.disabled = true;
      }
    } catch (err) {
      showError('加载文件夹列表失败: ' + err.message);
    } finally {
      loading.style.display = 'none';
    }
  }
  
  // 更新下拉列表选项
  function updateFolderOptions() {
    const selectedA = folderA.value;
    const selectedB = folderB.value;
    
    // 清空选项（保留默认选项）
    folderA.innerHTML = '<option value="">-- 请选择文件夹 A --</option>';
    folderB.innerHTML = '<option value="">-- 请选择文件夹 B --</option>';
    
    // 重新填充选项，排除对方已选择的
    folders.forEach(([path, display]) => {
      if (path !== selectedB) {
        const optA = document.createElement('option');
        optA.value = path;
        optA.textContent = display;
        if (path === selectedA) optA.selected = true;
        folderA.appendChild(optA);
      }
      
      if (path !== selectedA) {
        const optB = document.createElement('option');
        optB.value = path;
        optB.textContent = display;
        if (path === selectedB) optB.selected = true;
        folderB.appendChild(optB);
      }
    });
  }
  
  // 监听选择变化
  function updateStartButton() {
    const a = folderA.value;
    const b = folderB.value;
    startBtn.disabled = !a || !b || a === b;
    
    // 更新下拉列表，排除已选择的
    updateFolderOptions();
  }
  
  folderA.addEventListener('change', updateStartButton);
  folderB.addEventListener('change', updateStartButton);
  
  // 开始对比
  startBtn.addEventListener('click', async () => {
    const a = folderA.value;
    const b = folderB.value;
    
    loading.style.display = 'flex';
    try {
      const url = `/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`;
      const res = await fetch(url);
      const data = await res.json();
      
      if (data.error) {
        showError(data.error);
        return;
      }
      
      pairs = data.pairs;
      
      if (pairs.length === 0) {
        showError('这两个文件夹中没有同名图片');
        return;
      }
      
      labelA.textContent = '文件夹 A: ' + data.dirA;
      labelB.textContent = '文件夹 B: ' + data.dirB;
      
      selectScreen.style.display = 'none';
      compareScreen.style.display = 'flex';
      zoomControls.style.display = 'flex';
      loopControl.style.display = 'flex';
      openCompare(0);
    } catch (err) {
      showError('加载失败: ' + err.message);
    } finally {
      loading.style.display = 'none';
    }
  });
  
  function openCompare(index) {
    if (index < 0 || index >= pairs.length) return;
    currentIndex = index;
    const pair = pairs[index];
    
    // 重置缩放和平移
    pan = { x: 0, y: 0 };
    setZoom(1.0);
    
    const cachedA = preloadImage(pair.urlA);
    const cachedB = preloadImage(pair.urlB);
    imgA.src = cachedA.src;
    imgB.src = cachedB.src;
    
    compareTitle.textContent = pair.name;
    counter.textContent = `${index + 1} / ${pairs.length}`;
    
    // 预加载前后图片
    if (pairs.length > 1) {
      const prevIndex = (index - 1 + pairs.length) % pairs.length;
      const nextIndex = (index + 1) % pairs.length;
      preloadImage(pairs[prevIndex].urlA);
      preloadImage(pairs[prevIndex].urlB);
      preloadImage(pairs[nextIndex].urlA);
      preloadImage(pairs[nextIndex].urlB);
    }
  }
  
  function closeCompare() {
    compareScreen.style.display = 'none';
    zoomControls.style.display = 'none';
    loopControl.style.display = 'none';
    selectScreen.style.display = 'block';
    currentIndex = -1;
    pan = { x: 0, y: 0 };
    setZoom(1.0);
  }
  
  function showPrev() {
    if (loopEnabled) {
      openCompare((currentIndex - 1 + pairs.length) % pairs.length);
    } else {
      if (currentIndex > 0) {
        openCompare(currentIndex - 1);
      } else {
        showToast('已经是第一张了');
      }
    }
  }
  
  function showNext() {
    if (loopEnabled) {
      openCompare((currentIndex + 1) % pairs.length);
    } else {
      if (currentIndex < pairs.length - 1) {
        openCompare(currentIndex + 1);
      } else {
        showToast('已浏览完所有图片');
      }
    }
  }
  
  document.getElementById('back-btn').addEventListener('click', closeCompare);
  document.getElementById('prev-btn').addEventListener('click', showPrev);
  document.getElementById('next-btn').addEventListener('click', showNext);
  
  document.addEventListener('keydown', e => {
    if (compareScreen.style.display !== 'flex') return;
    if (e.key === 'Escape') closeCompare();
    else if (e.key === 'ArrowLeft') { e.preventDefault(); showPrev(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); showNext(); }
    else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomIn(); }
    else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomOut(); }
    else if (e.key === '0') { e.preventDefault(); zoomReset(); }
  });
  
  // 初始化
  loadFolders();
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
    global ROOT_DIR

    parser = argparse.ArgumentParser(description="局域网图片对比服务器")
    parser.add_argument("directory", nargs="?", default="D:/ResultCompare",
                        help="根目录路径（默认为当前目录）")
    parser.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    parser.add_argument("--host", default="0.0.0.0",
                        help="绑定地址（默认 0.0.0.0，允许局域网访问）")
    args = parser.parse_args()

    ROOT_DIR = Path(args.directory).resolve()
    if not ROOT_DIR.is_dir():
        print(f"错误: 文件夹不存在: {ROOT_DIR}")
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), CompareHandler)
    lan_ip = get_lan_ip()

    print("=" * 60)
    print("  📷 图片对比工具")
    print("=" * 60)
    print(f"  根目录: {ROOT_DIR}")
    print(f"  本机访问: http://127.0.0.1:{args.port}")
    print(f"  局域网访问: http://{lan_ip}:{args.port}")
    print("=" * 60)
    print("  使用说明:")
    print("  1. 在浏览器中打开上面的地址")
    print("  2. 从下拉菜单选择两个文件夹")
    print("  3. 点击'开始对比'按钮")
    print("  4. 使用方向键 ← → 切换图片")
    print("=" * 60)
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
