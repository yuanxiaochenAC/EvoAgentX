from typing import Dict, Any, Optional
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox   

class WorkFlowJSONEditorGUI:
    """GUI JSON Editor GUI based on tkinter"""
    
    def __init__(self, json_data: Dict[str, Any]):
        self.json_data = json_data
        self.result = None
        self.root = None
        
    def edit_json(self) -> Optional[Dict[str, Any]]:
        """启动JSON编辑器并返回修改后的数据"""
        try:
            import tkinter as tk
            from tkinter import ttk, scrolledtext, messagebox
        except ImportError:
            print("⚠️  tkinter不可用，使用文本编辑器")
            return self._edit_json_text()
        
        self.root = tk.Tk()
        self.root.title("WorkFlow JSON 编辑器")
        self.root.geometry("800x600")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="编辑 WorkFlow JSON 结构", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # 左侧按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.N), padx=(0, 10))
        
        # 按钮
        ttk.Button(button_frame, text="📝 格式化", command=self._format_json).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="✅ 验证", command=self._validate_json).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🔄 重置", command=self._reset_json).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="📋 复制", command=self._copy_json).pack(fill=tk.X, pady=2)
        
        ttk.Separator(button_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 快速操作按钮
        ttk.Label(button_frame, text="快速操作:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Button(button_frame, text="➕ 添加节点", command=self._add_node_quick).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🔗 添加边", command=self._add_edge_quick).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="📄 模板", command=self._insert_template).pack(fill=tk.X, pady=2)
        
        # 右侧文本编辑区域
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # 文本编辑器
        self.text_area = scrolledtext.ScrolledText(
            text_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=30,
            font=("Consolas", 10)
        )
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 插入JSON数据
        self.text_area.insert(tk.END, json.dumps(self.json_data, indent=2, ensure_ascii=False))
        
        # 底部按钮框架
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # 确认和取消按钮
        ttk.Button(bottom_frame, text="💾 保存并关闭", command=self._save_and_close).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(bottom_frame, text="❌ 取消", command=self._cancel).pack(side=tk.LEFT, padx=(0, 5))
        
        # 状态标签
        self.status_label = ttk.Label(bottom_frame, text="就绪", foreground="green")
        self.status_label.pack(side=tk.RIGHT)
        
        # 启动GUI
        self.root.mainloop()
        return self.result
    
    def _format_json(self):
        """格式化JSON"""
        try:
            text = self.text_area.get(1.0, tk.END)
            data = json.loads(text)
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, formatted)
            self.status_label.config(text="✅ 格式化完成", foreground="green")
        except json.JSONDecodeError as e:
            self.status_label.config(text=f"❌ JSON格式错误: {e}", foreground="red")
    
    def _validate_json(self):
        """验证JSON"""
        try:
            text = self.text_area.get(1.0, tk.END)
            data = json.loads(text)
            
            # 验证WorkFlow结构
            if not isinstance(data, dict):
                raise ValueError("根节点必须是字典")
            
            if 'nodes' not in data or not isinstance(data['nodes'], list):
                raise ValueError("必须包含nodes数组")
            
            node_names = set()
            for node in data['nodes']:
                if not isinstance(node, dict) or 'name' not in node:
                    raise ValueError("每个节点必须包含name字段")
                
                name = node['name']
                if name in node_names:
                    raise ValueError(f"节点名称重复: {name}")
                node_names.add(name)
            
            # 验证边
            if 'edges' in data:
                for edge in data['edges']:
                    if not isinstance(edge, dict):
                        continue
                    source = edge.get('source')
                    target = edge.get('target')
                    if source and source not in node_names:
                        raise ValueError(f"边的源节点不存在: {source}")
                    if target and target not in node_names:
                        raise ValueError(f"边的目标节点不存在: {target}")
            
            self.status_label.config(text="✅ JSON结构有效", foreground="green")
            
        except (json.JSONDecodeError, ValueError) as e:
            self.status_label.config(text=f"❌ 验证失败: {e}", foreground="red")
    
    def _reset_json(self):
        """重置JSON"""
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, json.dumps(self.json_data, indent=2, ensure_ascii=False))
        self.status_label.config(text="🔄 已重置", foreground="blue")
    
    def _copy_json(self):
        """复制JSON到剪贴板"""
        try:
            text = self.text_area.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="📋 已复制到剪贴板", foreground="blue")
        except Exception as e:
            self.status_label.config(text=f"❌ 复制失败: {e}", foreground="red")
    
    def _add_node_quick(self):
        """快速添加节点"""
        try:
            import tkinter.simpledialog as sd
            
            name = sd.askstring("添加节点", "节点名称:")
            if not name:
                return
            
            desc = sd.askstring("添加节点", "节点描述:")
            if not desc:
                desc = f"节点{name}的描述"
            
            node_template = {
                "class_name": "WorkFlowNode",
                "name": name,
                "description": desc,
                "inputs": [],
                "outputs": [],
                "agents": [],
                "status": "pending"
            }
            
            # 获取当前JSON
            current_text = self.text_area.get(1.0, tk.END)
            try:
                data = json.loads(current_text)
                data.setdefault('nodes', []).append(node_template)
                
                # 更新文本区域
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, json.dumps(data, indent=2, ensure_ascii=False))
                self.status_label.config(text=f"✅ 已添加节点: {name}", foreground="green")
                
            except json.JSONDecodeError:
                self.status_label.config(text="❌ 当前JSON格式错误，无法添加节点", foreground="red")
                
        except ImportError:
            self.status_label.config(text="❌ 无法使用对话框", foreground="red")
    
    def _add_edge_quick(self):
        """快速添加边"""
        try:
            import tkinter.simpledialog as sd
            
            # 获取当前节点列表
            current_text = self.text_area.get(1.0, tk.END)
            try:
                data = json.loads(current_text)
                nodes = data.get('nodes', [])
                node_names = [node.get('name') for node in nodes if node.get('name')]
                
                if len(node_names) < 2:
                    self.status_label.config(text="❌ 至少需要2个节点才能添加边", foreground="red")
                    return
                
                source = sd.askstring("添加边", f"源节点 (可选: {', '.join(node_names)}):")
                if not source or source not in node_names:
                    self.status_label.config(text="❌ 源节点无效", foreground="red")
                    return
                
                target = sd.askstring("添加边", f"目标节点 (可选: {', '.join(node_names)}):")
                if not target or target not in node_names:
                    self.status_label.config(text="❌ 目标节点无效", foreground="red")
                    return
                
                edge_template = {
                    "class_name": "WorkFlowEdge",
                    "source": source,
                    "target": target,
                    "priority": 0
                }
                
                data.setdefault('edges', []).append(edge_template)
                
                # 更新文本区域
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, json.dumps(data, indent=2, ensure_ascii=False))
                self.status_label.config(text=f"✅ 已添加边: {source} -> {target}", foreground="green")
                
            except json.JSONDecodeError:
                self.status_label.config(text="❌ 当前JSON格式错误，无法添加边", foreground="red")
                
        except ImportError:
            self.status_label.config(text="❌ 无法使用对话框", foreground="red")
    
    def _insert_template(self):
        """插入模板"""
        templates = {
            "简单节点": {
                "class_name": "WorkFlowNode",
                "name": "new_node",
                "description": "新节点描述",
                "inputs": [{"class_name": "Parameter", "name": "input1", "type": "string", "description": "输入参数", "required": True}],
                "outputs": [{"class_name": "Parameter", "name": "output1", "type": "string", "description": "输出参数", "required": True}],
                "agents": [],
                "status": "pending"
            },
            "CustomizeAgent": {
                "name": "my_agent",
                "description": "自定义Agent",
                "inputs": [{"name": "input1", "type": "string", "description": "输入", "required": True}],
                "outputs": [{"name": "output1", "type": "string", "description": "输出", "required": True}],
                "prompt": "处理输入：{input1}",
                "parse_mode": "str"
            }
        }
        
        # 创建模板选择窗口
        template_window = tk.Toplevel(self.root)
        template_window.title("选择模板")
        template_window.geometry("400x300")
        
        ttk.Label(template_window, text="选择要插入的模板:").pack(pady=10)
        
        template_listbox = tk.Listbox(template_window)
        template_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for template_name in templates.keys():
            template_listbox.insert(tk.END, template_name)
        
        def insert_selected():
            selection = template_listbox.curselection()
            if selection:
                template_name = template_listbox.get(selection[0])
                template_json = json.dumps(templates[template_name], indent=2, ensure_ascii=False)
                self.text_area.insert(tk.INSERT, f"\n{template_json}\n")
                self.status_label.config(text=f"✅ 已插入模板: {template_name}", foreground="green")
                template_window.destroy()
        
        ttk.Button(template_window, text="插入", command=insert_selected).pack(pady=10)
        ttk.Button(template_window, text="取消", command=template_window.destroy).pack()
    
    def _save_and_close(self):
        """保存并关闭"""
        try:
            text = self.text_area.get(1.0, tk.END)
            self.result = json.loads(text)
            self.root.destroy()
        except json.JSONDecodeError as e:
            self.status_label.config(text=f"❌ JSON格式错误: {e}", foreground="red")
    
    def _cancel(self):
        """取消"""
        self.result = None
        self.root.destroy()
    
    def _edit_json_text(self) -> Optional[Dict[str, Any]]:
        """使用文本编辑器编辑JSON（备用方案）"""
        import tempfile
        import subprocess
        import os
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(self.json_data, f, indent=2, ensure_ascii=False)
            temp_file = f.name
        
        try:
            print(f"📝 正在打开文件编辑器: {temp_file}")
            print("💡 编辑完成后请保存文件并关闭编辑器")
            
            # 根据操作系统选择编辑器
            if os.name == 'nt':  # Windows
                subprocess.run(['notepad', temp_file])
            elif os.name == 'posix':  # Linux/Mac
                subprocess.run(['nano', temp_file])
            
            # 读取编辑后的文件
            with open(temp_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            return result
            
        except Exception as e:
            print(f"❌ 编辑器打开失败: {e}")
            return None
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file)
            except:
                pass