"""
Real-time Inference and Visualization System
Separate Object and Action Recognition Models
实时物体和动作识别系统 - 双模型版本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from scipy import ndimage
import time
import json
import os
from PIL import Image
import sys
import platform

from serial_receiver_强力降噪版 import PressureSensorReceiver
from cnn_model import get_model

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure matplotlib to support Chinese characters and avoid font warnings
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False


class DualTaskRealtimeInference:
    """
    Real-time inference system with separate object and action recognition models
    支持两种模式切换：物体识别 | 动作识别
    """
    
    def __init__(self, object_model_path, action_model_path, receiver, device='cuda', mode='object'):
        """
        Args:
            object_model_path: Path to object recognition model
            action_model_path: Path to action recognition model
            receiver: PressureSensorReceiver instance
            device: 'cuda' or 'cpu'
            mode: 'object' (物体识别) 或 'action' (动作识别)
        """
        self.receiver = receiver
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Mode selection
        self.mode = mode  # 'object' or 'action'
        self.available_modes = ['object', 'action']
        
        # Labels
        self.object_labels = ['empty', 'ball', 'bottle', 'phone', 'spanner']
        self.action_labels = ['none', 'hold', 'tap', 'hammer', 'finger_press']
        
        # Load Object Recognition Model
        print(f"Loading object model from {object_model_path}")
        obj_checkpoint = torch.load(object_model_path, map_location=self.device)
        self.object_model = get_model('advanced', task='object', num_classes=len(self.object_labels))
        self.object_model.load_state_dict(obj_checkpoint['model_state_dict'])
        self.object_model.to(self.device)
        self.object_model.eval()
        
        # Load Action Recognition Model
        print(f"Loading action model from {action_model_path}")
        act_checkpoint = torch.load(action_model_path, map_location=self.device)
        self.action_model = get_model('advanced', task='action', num_classes=len(self.action_labels))
        self.action_model.load_state_dict(act_checkpoint['model_state_dict'])
        self.action_model.to(self.device)
        self.action_model.eval()
        
        print(f"Models loaded successfully. Using device: {self.device}")
        print(f"当前模式: {self._get_mode_name()}")
        print("提示: [1]=物体 [2]=动作 | [3-6]=热力图模式 [R]=设参考 [Q]=退出")
        
        # Prediction smoothing
        self.prediction_history = {
            'object': [],
            'action': []
        }
        self.history_length = 5
        
        # Performance tracking
        self.inference_times = []
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Load reference images
        self.reference_images = self._load_reference_images()
    
    def _get_mode_name(self):
        """获取模式名称"""
        mode_names = {
            'object': '物体识别模式',
            'action': '动作识别模式'
        }
        return mode_names.get(self.mode, '未知模式')
    
    def switch_mode(self, new_mode):
        """切换识别模式（通过键盘）"""
        if new_mode in self.available_modes:
            old_mode = self.mode
            self.mode = new_mode
            # 清空历史预测
            self.prediction_history = {'object': [], 'action': []}
            print(f"\n🔄 模式切换: {self._get_mode_name()}")
            # 更新按钮颜色（如果按钮已创建）
            if hasattr(self, 'btn_object'):
                self._update_button_colors()
            # 更新窗口标题
            if hasattr(self, 'fig'):
                title_text = f'压力传感器识别系统 - {self._get_mode_name()}'
                self.fig.suptitle(title_text, 
                                 fontsize=18, fontweight='bold', color='#00d9ff', y=0.98)
            return True
        return False
    
    def button_switch_mode(self, new_mode):
        """通过按钮切换模式"""
        if self.switch_mode(new_mode):
            print(f"✅ 已切换到: {self._get_mode_name()}")
    
    def _update_button_colors(self):
        """更新按钮颜色以突出显示当前激活的模式"""
        # 定义激活和未激活的颜色
        active_colors = {
            'object': '#27ae60',  # 绿色（激活）
            'action': '#f39c12'  # 橙色（激活）
        }
        inactive_colors = {
            'object': '#95a5a6',  # 灰色（未激活）
            'action': '#95a5a6'
        }
        
        # 更新物体识别按钮
        if self.mode == 'object':
            self.btn_object.color = active_colors['object']
            self.btn_object.hovercolor = '#229954'
        else:
            self.btn_object.color = inactive_colors['object']
            self.btn_object.hovercolor = '#7f8c8d'
        self.btn_object.ax.set_facecolor(self.btn_object.color)
        
        # 更新动作识别按钮
        if self.mode == 'action':
            self.btn_action.color = active_colors['action']
            self.btn_action.hovercolor = '#e67e22'
        else:
            self.btn_action.color = inactive_colors['action']
            self.btn_action.hovercolor = '#7f8c8d'
        self.btn_action.ax.set_facecolor(self.btn_action.color)
        
        # 刷新显示
        self.fig.canvas.draw_idle()
    
    def _load_reference_images(self):
        """动态加载参考图片"""
        images = {}
        ref_dir = 'reference_images'
        
        print(f"\n正在加载参考图片...")
        print(f"参考图片目录: {ref_dir}")
        
        if not os.path.exists(ref_dir):
            print(f"❌ 警告: 未找到参考图片目录 {ref_dir}")
            return images
        
        print(f"✓ 找到参考图片目录")
        
        # 加载物品图片（从 objects 子目录）
        print(f"\n加载物品图片:")
        objects_dir = os.path.join(ref_dir, 'objects')
        if os.path.exists(objects_dir):
            for obj in self.object_labels:
                img_path = os.path.join(objects_dir, f'{obj}.png')
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        images[f'object_{obj}'] = img_array
                        print(f"  ✓ {img_path} - 尺寸: {img_array.shape}")
                    except Exception as e:
                        print(f"  ❌ 加载失败 {img_path}: {e}")
                else:
                    print(f"  ⚠ 未找到 {img_path}")
        
        # 加载动作图片（从 actions 子目录）
        print(f"\n加载动作图片:")
        actions_dir = os.path.join(ref_dir, 'actions')
        if os.path.exists(actions_dir):
            for act in self.action_labels:
                img_path = os.path.join(actions_dir, f'{act}.png')
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        images[f'action_{act}'] = img_array
                        print(f"  ✓ {img_path} - 尺寸: {img_array.shape}")
                    except Exception as e:
                        print(f"  ❌ 加载失败 {img_path}: {e}")
                else:
                    print(f"  ⚠ 未找到 {img_path}")
        
        print(f"\n✓ 成功加载 {len(images)} 张参考图片")
        return images
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for model input
        
        Args:
            frame: numpy array (16, 16) with uint8 values
            
        Returns:
            torch.Tensor: (1, 1, 16, 16) normalized tensor
        """
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        frame = np.expand_dims(frame, axis=(0, 1))
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).to(self.device)
        
        return frame_tensor
    
    def predict(self, frame):
        """
        Run inference on a single frame (根据当前模式运行对应的模型)
        
        Args:
            frame: numpy array (16, 16)
            
        Returns:
            dict: Predictions with probabilities
        """
        start_time = time.time()
        
        # Preprocess
        frame_tensor = self.preprocess_frame(frame)
        
        # Inference - 根据模式决定运行哪个模型
        result = {}
        
        with torch.no_grad():
            # Object recognition (在 object 模式下运行)
            if self.mode == 'object':
                obj_out = self.object_model(frame_tensor)
                obj_probs = torch.softmax(obj_out, dim=1)[0]
                obj_pred = torch.argmax(obj_probs).item()
                
                # Smooth predictions with history
                self.prediction_history['object'].append(obj_pred)
                if len(self.prediction_history['object']) > self.history_length:
                    self.prediction_history['object'].pop(0)
                
                # Use most common prediction in history (voting)
                obj_smooth = max(set(self.prediction_history['object']), 
                                key=self.prediction_history['object'].count)
                
                result['object'] = {
                    'label': self.object_labels[obj_smooth],
                    'confidence': obj_probs[obj_pred].item(),
                    'probabilities': obj_probs.cpu().numpy()
                }
            
            # Action recognition (在 action 模式下运行)
            if self.mode == 'action':
                act_out = self.action_model(frame_tensor)
                act_probs = torch.softmax(act_out, dim=1)[0]
                act_pred = torch.argmax(act_probs).item()
                
                # Smooth predictions with history
                self.prediction_history['action'].append(act_pred)
                if len(self.prediction_history['action']) > self.history_length:
                    self.prediction_history['action'].pop(0)
                
                # Use most common prediction in history (voting)
                act_smooth = max(set(self.prediction_history['action']), 
                                key=self.prediction_history['action'].count)
                
                result['action'] = {
                    'label': self.action_labels[act_smooth],
                    'confidence': act_probs[act_pred].item(),
                    'probabilities': act_probs.cpu().numpy()
                }
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        result['inference_time'] = inference_time
        return result
    
    def start_visualization(self):
        """启动可视化"""
        plt.style.use('dark_background')
        
        # 使用更高的DPI和更大的尺寸提升整体显示清晰度
        self.fig = plt.figure(figsize=(18, 10), facecolor='#0a0e27', dpi=110)
        title_text = f'压力传感器识别系统 - {self._get_mode_name()}'
        self.fig.suptitle(title_text, 
                         fontsize=18, fontweight='bold', color='#00d9ff', y=0.98)
        
        # 重新设计布局：左边2列（压力图和参考图），右边3列（结果和统计）
        gs = self.fig.add_gridspec(3, 5, hspace=0.35, wspace=0.35,
                                   left=0.04, right=0.98, top=0.94, bottom=0.13,
                                   width_ratios=[1.2, 1.2, 1, 1, 1])
        
        # === 左上：压力热力图 - 高分辨率256x256 ===
        self.ax_pressure = self.fig.add_subplot(gs[0:2, 0:2])
        
        # 创建自定义醒目的colormap（黑->蓝->青->绿->黄->橙->红->白）
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = [
            (0.0, '#000000'),  # 黑色（无压力）
            (0.15, '#0000FF'), # 深蓝
            (0.30, '#00FFFF'), # 青色
            (0.45, '#00FF00'), # 绿色
            (0.60, '#FFFF00'), # 黄色
            (0.75, '#FF8800'), # 橙色
            (0.90, '#FF0000'), # 红色
            (1.0, '#FFFFFF')   # 白色（最大压力）
        ]
        positions = [c[0] for c in colors_list]
        colors = [c[1] for c in colors_list]
        custom_cmap = LinearSegmentedColormap.from_list('pressure_enhanced', 
                                                        list(zip(positions, colors)), N=256)
        
        # Visualization mode: 'absolute', 'dynamic', 'difference', 'threshold'
        self.viz_mode = 'dynamic'  # Default mode
        self.reference_frame = None  # For difference mode
        self.display_threshold = 10  # Threshold for threshold mode
        self.recent_frames = []  # Store recent frames for smooth dynamic range
        self.max_recent_frames = 10
        
        self.im_pressure = self.ax_pressure.imshow(np.zeros((256, 256)), 
                                                    cmap=custom_cmap, vmin=0, vmax=255,
                                                    interpolation='bilinear',
                                                    aspect='equal')
        
        # Dynamic title based on mode
        self.pressure_title = self.ax_pressure.set_title(
            '实时压力分布 [DYNAMIC MODE]', fontsize=13, 
            fontweight='bold', color='#00d9ff', pad=8)
        
        # 添加网格显示256个格子
        self.ax_pressure.set_xticks(np.arange(0, 256, 16))
        self.ax_pressure.set_yticks(np.arange(0, 256, 16))
        self.ax_pressure.grid(True, color='#333333', linewidth=0.5, alpha=0.3)
        self.ax_pressure.tick_params(colors='#00d9ff', labelsize=7)
        self.ax_pressure.spines['top'].set_visible(False)
        self.ax_pressure.spines['right'].set_visible(False)
        self.ax_pressure.spines['bottom'].set_visible(False)
        self.ax_pressure.spines['left'].set_visible(False)
        self.cbar = plt.colorbar(self.im_pressure, ax=self.ax_pressure, 
                                fraction=0.046, pad=0.04)
        self.cbar.set_label('压力值', rotation=270, labelpad=15, 
                           fontsize=10, color='#00d9ff')
        self.cbar.ax.tick_params(colors='#00d9ff', labelsize=9)
        self.cbar.outline.set_visible(False)
        
        # === 左下：参考图片（根据模式显示物体或动作参考图）===
        self.ax_ref = self.fig.add_subplot(gs[2, 0:2])
        self.ax_ref.set_facecolor('#0f1535')
        self.ax_ref.set_xticks([])
        self.ax_ref.set_yticks([])
        self.ax_ref.set_title('参考图', fontsize=12, 
                              fontweight='bold', color='#00d9ff', pad=8)
        for spine in self.ax_ref.spines.values():
            spine.set_visible(False)
        placeholder = np.ones((200, 200, 3), dtype=np.uint8) * 20
        self.im_ref = self.ax_ref.imshow(placeholder, aspect='auto', 
                                         interpolation='lanczos',
                                         resample=True,
                                         filternorm=True)
        self.ax_ref.set_aspect('equal')
        self.text_ref_label = self.ax_ref.text(
            0.5, -0.06, '', transform=self.ax_ref.transAxes,
            fontsize=12, ha='center', va='top', color='#00d9ff', fontweight='bold'
        )
        
        # === 右上：主结果卡片 ===
        self.ax_result = self.fig.add_subplot(gs[0, 2:5])
        self.ax_result.set_xlim(0, 1)
        self.ax_result.set_ylim(0, 1)
        self.ax_result.axis('off')
        self.ax_result.set_facecolor('#0f1535')
        
        self.text_main_result = self.ax_result.text(
            0.5, 0.65, '', fontsize=20, fontweight='bold',
            ha='center', va='center', color='#00ff88'
        )
        self.text_scene_type = self.ax_result.text(
            0.5, 0.3, '', fontsize=12,
            ha='center', va='center', color='#00d9ff'
        )
        
        # === 右上下：性能统计 ===
        self.ax_stats = self.fig.add_subplot(gs[1, 2:5])
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor('#0f1535')
        
        self.ax_stats.text(0.5, 0.88, '性能统计', fontsize=12,
                          ha='center', va='center', color='#00d9ff',
                          fontweight='bold')
        
        self.text_stats = self.ax_stats.text(
            0.15, 0.45, '', fontsize=10, ha='left', va='center',
            color='#ffffff', linespacing=1.8
        )
        
        # === 右中：物品识别条形图 ===
        self.ax_object = self.fig.add_subplot(gs[2, 2:5])
        self.ax_object.set_facecolor('#0f1535')
        self.bars_object = self.ax_object.barh(
            range(len(self.object_labels)), [0]*len(self.object_labels),
            color='#00d9ff', alpha=0.8
        )
        self.ax_object.set_yticks(range(len(self.object_labels)))
        self.ax_object.set_yticklabels(self.object_labels, fontsize=11, color='#ffffff')
        self.ax_object.set_xlim([0, 1])
        self.ax_object.set_xlabel('置信度', fontsize=10, color='#00d9ff')
        self.ax_object.set_title('物品识别', fontsize=12, 
                                fontweight='bold', color='#00d9ff', pad=8)
        self.ax_object.grid(axis='x', alpha=0.3, color='#00d9ff', linestyle='--', linewidth=0.8)
        self.ax_object.tick_params(colors='#00d9ff', labelsize=9)
        for spine in self.ax_object.spines.values():
            spine.set_visible(False)
        
        # === 右下：动作识别条形图 ===
        self.ax_action = self.fig.add_subplot(gs[2, 2:5])  # 与物品识别共享同一位置
        self.ax_action.set_facecolor('#0f1535')
        self.bars_action = self.ax_action.barh(
            range(len(self.action_labels)), [0]*len(self.action_labels),
            color='#ff6b35', alpha=0.8
        )
        self.ax_action.set_yticks(range(len(self.action_labels)))
        self.ax_action.set_yticklabels(self.action_labels, fontsize=11, color='#ffffff')
        self.ax_action.set_xlim([0, 1])
        self.ax_action.set_xlabel('置信度', fontsize=10, color='#ff6b35')
        self.ax_action.set_title('动作识别', fontsize=12, 
                                fontweight='bold', color='#ff6b35', pad=8)
        self.ax_action.grid(axis='x', alpha=0.3, color='#ff6b35', linestyle='--', linewidth=0.8)
        self.ax_action.tick_params(colors='#ff6b35', labelsize=9)
        for spine in self.ax_action.spines.values():
            spine.set_visible(False)
        
        # === 底部：模式选择按钮区域 ===
        button_y = 0.03  # 按钮Y位置（降低以腾出空间给可视化按钮）
        button_height = 0.045
        button_width = 0.18
        gap = 0.035
        
        # === 可视化模式按钮（顶部一行）===
        viz_button_y = 0.09
        viz_button_width = 0.09
        viz_button_height = 0.038
        viz_gap = 0.008
        
        # 添加分隔线
        separator_line = plt.Line2D([0.04, 0.98], [0.14, 0.14], 
                                    transform=self.fig.transFigure, 
                                    color='#00d9ff', linewidth=2, linestyle='--', alpha=0.5)
        self.fig.add_artist(separator_line)
        
        # 可视化模式标签
        self.fig.text(0.06, viz_button_y + viz_button_height + 0.01, 
                     '热力图模式', fontsize=11, fontweight='bold', 
                     color='#00ff88', ha='left', va='bottom')
        
        # Absolute Mode Button
        ax_viz_abs = plt.axes([0.20, viz_button_y, viz_button_width, viz_button_height])
        self.btn_viz_abs = Button(ax_viz_abs, 'Absolute', 
                                  color='#34495e', hovercolor='#4a5f7f')
        self.btn_viz_abs.label.set_fontsize(8)
        self.btn_viz_abs.label.set_fontweight('bold')
        self.btn_viz_abs.label.set_color('white')
        self.btn_viz_abs.on_clicked(lambda event: self.switch_viz_mode('absolute'))
        
        # Dynamic Mode Button (default selected)
        ax_viz_dyn = plt.axes([0.20 + viz_button_width + viz_gap, viz_button_y, viz_button_width, viz_button_height])
        self.btn_viz_dyn = Button(ax_viz_dyn, 'Dynamic', 
                                  color='#27ae60', hovercolor='#229954')
        self.btn_viz_dyn.label.set_fontsize(8)
        self.btn_viz_dyn.label.set_fontweight('bold')
        self.btn_viz_dyn.label.set_color('white')
        self.btn_viz_dyn.on_clicked(lambda event: self.switch_viz_mode('dynamic'))
        
        # Difference Mode Button
        ax_viz_diff = plt.axes([0.20 + 2*(viz_button_width + viz_gap), viz_button_y, viz_button_width, viz_button_height])
        self.btn_viz_diff = Button(ax_viz_diff, 'Difference', 
                                   color='#34495e', hovercolor='#4a5f7f')
        self.btn_viz_diff.label.set_fontsize(8)
        self.btn_viz_diff.label.set_fontweight('bold')
        self.btn_viz_diff.label.set_color('white')
        self.btn_viz_diff.on_clicked(lambda event: self.switch_viz_mode('difference'))
        
        # Threshold Mode Button
        ax_viz_thresh = plt.axes([0.20 + 3*(viz_button_width + viz_gap), viz_button_y, viz_button_width, viz_button_height])
        self.btn_viz_thresh = Button(ax_viz_thresh, 'Threshold', 
                                     color='#34495e', hovercolor='#4a5f7f')
        self.btn_viz_thresh.label.set_fontsize(8)
        self.btn_viz_thresh.label.set_fontweight('bold')
        self.btn_viz_thresh.label.set_color('white')
        self.btn_viz_thresh.on_clicked(lambda event: self.switch_viz_mode('threshold'))
        
        # Set Reference Button (for difference mode)
        ax_set_ref = plt.axes([0.20 + 4*(viz_button_width + viz_gap), viz_button_y, viz_button_width, viz_button_height])
        self.btn_set_ref = Button(ax_set_ref, 'Set Ref', 
                                  color='#8e44ad', hovercolor='#7d3c98')
        self.btn_set_ref.label.set_fontsize(8)
        self.btn_set_ref.label.set_fontweight('bold')
        self.btn_set_ref.label.set_color('white')
        self.btn_set_ref.on_clicked(lambda event: self.set_reference_frame())
        
        # 添加识别模式选择标签
        self.fig.text(0.06, button_y + button_height + 0.012, 
                     '识别模式', fontsize=12, fontweight='bold', 
                     color='#00d9ff', ha='left', va='bottom')
        
        # 物体识别按钮
        ax_btn_object = plt.axes([0.25, button_y, button_width, button_height])
        self.btn_object = Button(ax_btn_object, '物体识别', 
                                 color='#e74c3c', hovercolor='#c0392b')
        self.btn_object.label.set_fontsize(11)
        self.btn_object.label.set_fontweight('bold')
        self.btn_object.label.set_color('white')
        self.btn_object.on_clicked(lambda event: self.button_switch_mode('object'))
        
        # 动作识别按钮
        ax_btn_action = plt.axes([0.25 + button_width + gap, button_y, button_width, button_height])
        self.btn_action = Button(ax_btn_action, '动作识别', 
                                color='#3498db', hovercolor='#2980b9')
        self.btn_action.label.set_fontsize(11)
        self.btn_action.label.set_fontweight('bold')
        self.btn_action.label.set_color('white')
        self.btn_action.on_clicked(lambda event: self.button_switch_mode('action'))
        
        # 初始化按钮颜色状态
        self._update_button_colors()
        
        # 根据初始模式设置条形图可见性
        if self.mode == 'object':
            self.ax_object.set_visible(True)
            self.ax_action.set_visible(False)
        elif self.mode == 'action':
            self.ax_object.set_visible(False)
            self.ax_action.set_visible(True)
        
        # 添加键盘事件处理
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Start receiver
        self.receiver.start_continuous_reading()
        
        # Start animation
        self.ani_running = True
        plt.show(block=False)
        
        self.update_loop()
    
    def _process_frame_for_display(self, frame):
        """
        Process frame based on visualization mode
        Returns: (processed_frame, vmin, vmax)
        """
        if self.viz_mode == 'absolute':
            # Fixed range 0-255
            return frame, 0, 255
        
        elif self.viz_mode == 'dynamic':
            # Dynamic range with percentile clipping and temporal smoothing
            if len(self.recent_frames) > 0:
                # Use recent frames for smoother range adjustment
                stacked = np.array(self.recent_frames)
                p2 = np.percentile(stacked, 2)
                p98 = np.percentile(stacked, 98)
            else:
                p2 = np.percentile(frame, 2)
                p98 = np.percentile(frame, 98)
            
            # Ensure minimum range for visibility
            if p98 - p2 < 10:
                center = (p98 + p2) / 2
                p2 = max(0, center - 5)
                p98 = min(255, center + 5)
            
            return frame, p2, p98
        
        elif self.viz_mode == 'difference':
            # Show difference from reference frame
            if self.reference_frame is None:
                # Auto-set reference if not set
                self.reference_frame = frame.copy()
                print("🔧 Auto-set reference frame")
            
            # Calculate difference (can be negative)
            diff = frame.astype(np.float32) - self.reference_frame.astype(np.float32)
            
            # Use symmetric range around 0
            max_abs = max(abs(np.min(diff)), abs(np.max(diff)))
            if max_abs < 10:
                max_abs = 10  # Minimum range
            
            # Shift to positive range for display
            diff_display = diff + max_abs
            
            return diff_display, 0, 2 * max_abs
        
        elif self.viz_mode == 'threshold':
            # Only show values above threshold
            thresholded = frame.copy()
            thresholded[thresholded < self.display_threshold] = 0
            
            # Dynamic range for non-zero values
            non_zero = thresholded[thresholded > 0]
            if len(non_zero) > 0:
                vmax = np.percentile(non_zero, 98)
            else:
                vmax = 255
            
            return thresholded, 0, vmax
        
        else:
            # Fallback to absolute
            return frame, 0, 255
    
    def switch_viz_mode(self, new_mode):
        """Switch visualization mode"""
        if self.viz_mode == new_mode:
            return
        
        self.viz_mode = new_mode
        
        # Update button colors
        self.btn_viz_abs.color = '#27ae60' if new_mode == 'absolute' else '#34495e'
        self.btn_viz_dyn.color = '#27ae60' if new_mode == 'dynamic' else '#34495e'
        self.btn_viz_diff.color = '#27ae60' if new_mode == 'difference' else '#34495e'
        self.btn_viz_thresh.color = '#27ae60' if new_mode == 'threshold' else '#34495e'
        
        # Update button axes colors
        self.btn_viz_abs.ax.set_facecolor(self.btn_viz_abs.color)
        self.btn_viz_dyn.ax.set_facecolor(self.btn_viz_dyn.color)
        self.btn_viz_diff.ax.set_facecolor(self.btn_viz_diff.color)
        self.btn_viz_thresh.ax.set_facecolor(self.btn_viz_thresh.color)
        
        # Update title
        mode_names = {
            'absolute': 'ABSOLUTE (0-255)',
            'dynamic': 'DYNAMIC (自适应)',
            'difference': 'DIFFERENCE (差异)',
            'threshold': f'THRESHOLD (>{self.display_threshold})'
        }
        
        # Auto-set reference for difference mode
        if new_mode == 'difference' and self.reference_frame is None and hasattr(self, 'current_frame'):
            self.reference_frame = self.current_frame.copy()
            print("🔧 Auto-set reference frame for difference mode")
        
        self.fig.canvas.draw_idle()
        print(f"🎨 切换到 {new_mode.upper()} 可视化模式")
    
    def set_reference_frame(self):
        """Set current frame as reference for difference mode"""
        if hasattr(self, 'current_frame'):
            self.reference_frame = self.current_frame.copy()
            print("✅ Reference frame updated!")
            print(f"   参考帧统计 - Min: {np.min(self.reference_frame):.1f}, "
                  f"Max: {np.max(self.reference_frame):.1f}, "
                  f"Mean: {np.mean(self.reference_frame):.1f}")
        else:
            print("❌ No frame available to set as reference")
    
    def on_key_press(self, event):
        """处理键盘事件"""
        if event.key == '1':
            self.switch_mode('object')
        elif event.key == '2':
            self.switch_mode('action')
        elif event.key == 'q':
            print("\n退出程序...")
            self.ani_running = False
            plt.close(self.fig)
        # Visualization mode shortcuts
        elif event.key == '3':
            self.switch_viz_mode('absolute')
        elif event.key == '4':
            self.switch_viz_mode('dynamic')
        elif event.key == '5':
            self.switch_viz_mode('difference')
        elif event.key == '6':
            self.switch_viz_mode('threshold')
        elif event.key == 'r':
            self.set_reference_frame()
    
    def update_loop(self):
        """Main update loop"""
        while self.ani_running and plt.fignum_exists(self.fig.number):
            frame_data = self.receiver.get_latest_frame(timeout=0.1)
            
            if frame_data:
                frame = frame_data['frame']
                self.frame_count += 1
                
                # Store frame for temporal smoothing and capture
                self.current_frame = frame.copy()
                self.recent_frames.append(frame.copy())
                if len(self.recent_frames) > self.max_recent_frames:
                    self.recent_frames.pop(0)
                
                # Run inference
                predictions = self.predict(frame)
                
                # Process frame based on visualization mode
                processed_frame, vmin, vmax = self._process_frame_for_display(frame)
                
                # 高质量插值到256x256显示
                frame_display = self.receiver.upscale_frame(processed_frame, target_size=256)
                
                # Update pressure map with processed high-res display
                self.im_pressure.set_data(frame_display)
                
                # Update colorbar limits dynamically
                self.im_pressure.set_clim(vmin=vmin, vmax=vmax)
                
                # 根据模式更新不同的显示内容
                obj_label = predictions.get('object', {}).get('label', 'N/A')
                act_label = predictions.get('action', {}).get('label', 'N/A')
                
                # 根据模式更新参考图
                if self.mode == 'object' and 'object' in predictions:
                    # 物体识别模式：显示物体参考图
                    obj_img_key = f'object_{obj_label}'
                    if obj_img_key in self.reference_images:
                        img = self.reference_images[obj_img_key]
                        self.im_ref.set_data(img)
                        self.im_ref.set_extent([0, img.shape[1], img.shape[0], 0])
                        self.ax_ref.set_xlim([0, img.shape[1]])
                        self.ax_ref.set_ylim([img.shape[0], 0])
                        self.text_ref_label.set_text(f"物品: {obj_label.upper()}")
                        self.ax_ref.set_title('物体参考图', fontsize=12, 
                                             fontweight='bold', color='#00d9ff', pad=8)
                elif self.mode == 'action' and 'action' in predictions:
                    # 动作识别模式：显示动作参考图
                    act_img_key = f'action_{act_label}'
                    if act_img_key in self.reference_images:
                        img = self.reference_images[act_img_key]
                        self.im_ref.set_data(img)
                        self.im_ref.set_extent([0, img.shape[1], img.shape[0], 0])
                        self.ax_ref.set_xlim([0, img.shape[1]])
                        self.ax_ref.set_ylim([img.shape[0], 0])
                        self.text_ref_label.set_text(f"动作: {act_label.upper()}")
                        self.ax_ref.set_title('动作参考图', fontsize=12, 
                                             fontweight='bold', color='#ff6b35', pad=8)
                
                # Update main result - 根据模式显示不同内容
                if self.mode == 'object':
                    main_text = f"识别物品\n\n{obj_label.upper()}"
                    scene_text = "物体识别模式"
                elif self.mode == 'action':
                    main_text = f"识别动作\n\n{act_label.upper()}"
                    scene_text = "动作识别模式"
                
                self.text_main_result.set_text(main_text)
                self.text_scene_type.set_text(scene_text)
                
                # Update statistics
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
                
                stats_lines = []
                if 'object' in predictions:
                    stats_lines.append(f"物品: {obj_label} ({predictions['object']['confidence']*100:.1f}%)")
                if 'action' in predictions:
                    stats_lines.append(f"动作: {act_label} ({predictions['action']['confidence']*100:.1f}%)")
                stats_lines.append(f"FPS: {self.fps:.1f}")
                stats_lines.append(f"推理时间: {avg_inference_time*1000:.1f} ms")
                stats_lines.append(f"总帧数: {self.frame_count}")
                stats_lines.append(f"压力范围: {vmin:.0f}-{vmax:.0f}")
                stats_lines.append(f"\n识别模式: {self._get_mode_name()}")
                
                # Add visualization mode info
                viz_mode_names = {
                    'absolute': '绝对值',
                    'dynamic': '动态',
                    'difference': '差异',
                    'threshold': '阈值'
                }
                stats_lines.append(f"可视化: {viz_mode_names.get(self.viz_mode, '未知')}")
                
                self.text_stats.set_text('\n'.join(stats_lines))
                
                # 根据模式显示/隐藏条形图
                if self.mode == 'object':
                    # 物体识别模式：显示物品条形图，隐藏动作条形图
                    self.ax_object.set_visible(True)
                    self.ax_action.set_visible(False)
                    if 'object' in predictions:
                        obj_probs = predictions['object']['probabilities']
                        for bar, prob in zip(self.bars_object, obj_probs):
                            bar.set_width(prob)
                            if prob == max(obj_probs):
                                bar.set_color('#00ff88')
                                bar.set_alpha(1.0)
                            else:
                                bar.set_color('#00d9ff')
                                bar.set_alpha(0.6)
                elif self.mode == 'action':
                    # 动作识别模式：显示动作条形图，隐藏物品条形图
                    self.ax_object.set_visible(False)
                    self.ax_action.set_visible(True)
                    if 'action' in predictions:
                        act_probs = predictions['action']['probabilities']
                        for bar, prob in zip(self.bars_action, act_probs):
                            bar.set_width(prob)
                            if prob == max(act_probs):
                                bar.set_color('#ff6b35')
                                bar.set_alpha(1.0)
                            else:
                                bar.set_color('#ffaa00')
                                bar.set_alpha(0.6)
                
                # Update pressure map title with visualization mode
                viz_mode_names_short = {
                    'absolute': 'ABS',
                    'dynamic': 'DYN',
                    'difference': 'DIFF',
                    'threshold': 'THR'
                }
                viz_short = viz_mode_names_short.get(self.viz_mode, 'UNK')
                
                if self.mode == 'object':
                    title_text = f"实时压力分布 [{viz_short}] - 物品: {obj_label.upper()}"
                elif self.mode == 'action':
                    title_text = f"实时压力分布 [{viz_short}] - 动作: {act_label.upper()}"
                
                self.ax_pressure.set_title(
                    title_text,
                    fontsize=13, fontweight='bold', color='#00d9ff', pad=8
                )
                
                plt.pause(0.001)
        
        # Cleanup
        self.receiver.stop_continuous_reading()
        plt.close()


def main():
    """Main function"""
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='压力传感器实时推理系统')
    parser.add_argument('--mode', type=str, default='object', 
                       choices=['object', 'action'],
                       help='启动模式: object(物体识别) 或 action(动作识别)')
    parser.add_argument('--port', type=str, default='COM3',
                       help='串口号 (默认: COM3)')
    args = parser.parse_args()
    
    OBJECT_MODEL_PATH = 'models/best_object_model.pth'
    ACTION_MODEL_PATH = 'models/best_action_model.pth'
    PORT = args.port
    START_MODE = args.mode
    
    # Check if models exist
    if not os.path.exists(OBJECT_MODEL_PATH):
        print(f"❌ 错误: 找不到物体模型 {OBJECT_MODEL_PATH}")
        print("   请先运行 train.py 训练模型")
        return
    
    if not os.path.exists(ACTION_MODEL_PATH):
        print(f"❌ 错误: 找不到动作模型 {ACTION_MODEL_PATH}")
        print("   请先运行 train.py 训练模型")
        return
    
    # Create receiver with strong noise reduction (针对高帧率优化)
    receiver = PressureSensorReceiver(
        port=PORT,
        enable_noise_reduction=True,
        noise_threshold=15,  # 适中阈值
        temporal_smoothing=0.7,  # 增强时域平滑（从0.6提高到0.7）
        spatial_smoothing=0.8,  # 增强空间平滑（从0.7提高到0.8）
        multi_frame_average=True,  # 启用多帧平均
        average_frames=10  # 增加到10帧（从6提高到10），大幅降低噪声
    )
    
    # Connect
    if not receiver.connect():
        print("Failed to connect to serial port!")
        print("Please check:")
        print("  1. Serial port is correct")
        print("  2. FPGA is powered on")
        print("  3. No other program is using the port")
        return
    
    # 背景校准
    print("\n" + "="*70)
    print("🔧 实时推理前需要先进行背景校准")
    print("="*70)
    print("⚠️  重要提示:")
    print("   1. 请确保传感器表面干净,没有任何物体")
    print("   2. 确保传感器处于稳定状态")
    print("   3. 背景校准需要约1秒钟")
    print("="*70)
    input("\n✋ 准备好后按回车键开始背景校准...")
    
    receiver.calibrate_background(num_samples=20)
    
    print("\n✅ 背景校准完成! 准备启动实时推理系统...")
    print("="*70 + "\n")
    
    # Create inference system
    inference_system = DualTaskRealtimeInference(
        OBJECT_MODEL_PATH, 
        ACTION_MODEL_PATH, 
        receiver, 
        device='cuda',
        mode=START_MODE  # 使用命令行指定的模式
    )
    
    # Start visualization
    print("\n" + "="*70)
    print("启动实时推理系统...")
    print("="*70)
    print(f"  当前模式: {inference_system._get_mode_name()}")
    print(f"  可视化模式: DYNAMIC (动态自适应)")
    print("\n  识别模式切换:")
    print("    方法1: 点击界面底部的 '识别模式' 按钮")
    print("           物体识别 | 动作识别")
    print("    方法2: 使用键盘快捷键")
    print("           [1] 键 → 物体识别")
    print("           [2] 键 → 动作识别")
    print("\n  可视化模式切换:")
    print("    方法1: 点击界面的 '热力图模式' 按钮")
    print("           Absolute | Dynamic | Difference | Threshold | Set Ref")
    print("    方法2: 使用键盘快捷键")
    print("           [3] → Absolute (固定0-255)")
    print("           [4] → Dynamic (自适应范围) ⭐推荐")
    print("           [5] → Difference (差异显示) ⭐解决全图抖动")
    print("           [6] → Threshold (阈值过滤)")
    print("           [R] → 更新参考帧 (用于Difference模式)")
    print("\n  其他快捷键:")
    print("           [Q] 键 → 退出程序")
    print("\n  提示:")
    print("    - 物体识别: 识别传感器上是什么物品")
    print("    - 动作识别: 识别正在执行什么动作")
    print("    - Dynamic模式: colorbar自动调整范围")
    print("    - Difference模式: 只显示变化区域")
    print("    - 激活的按钮会高亮显示为绿色")
    print("="*70 + "\n")
    
    try:
        inference_system.start_visualization()
    except KeyboardInterrupt:
        print("\n退出...")
    finally:
        receiver.disconnect()


if __name__ == '__main__':
    main()
