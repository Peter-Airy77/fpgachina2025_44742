"""
Data Collection and Annotation Tool
Collects pressure sensor data with labels for training CNN models

IMPORTANT: This tool collects data for TWO SEPARATE TASKS:
- Object Recognition Mode: Collects ONLY object labels (ball, bottle, empty, pen, phone, hand)
- Action Recognition Mode: Collects ONLY action labels (none, press, rotate, tap, touch, hold)

These are INDEPENDENT tasks, trained with separate models.
"""

import numpy as np
import time
import os
import json
import h5py
from datetime import datetime
from serial_receiver_强力降噪版 import PressureSensorReceiver
from PIL import Image, ImageDraw, ImageFont

# Set interactive backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for better Windows compatibility
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# 配置中文字体支持，避免字体警告
import platform
if platform.system() == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
elif platform.system() == 'Darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
else:  # Linux
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class DataCollector:
    """
    Interactive data collection tool for labeling pressure sensor data
    """
    
    def __init__(self, receiver, save_dir='data/collected', mode='object'):
        """
        Initialize data collector with dynamic mode switching
        
        Args:
            receiver: PressureSensorReceiver instance
            save_dir: Directory to save collected data
            mode: 'object' or 'action' - initial collection mode
        """
        self.receiver = receiver
        self.save_dir = save_dir
        self.mode = mode  # 'object' or 'action' - can be switched dynamically
        os.makedirs(save_dir, exist_ok=True)
        
        # Data storage
        self.collected_data = []
        
        # Labels configuration
        self.object_labels = ['empty', 'ball', 'bottle', 'phone', 'spanner']
        self.action_labels = ['none', 'hold', 'tap', 'hammer', 'finger_press']
        
        # Chinese label mapping (中文标签映射)
        self.object_labels_cn = {
            'empty': '空',
            'ball': '球',
            'bottle': '瓶子',
            'phone': '手机',
            'spanner': '扳手'
        }
        self.action_labels_cn = {
            'none': '无',
            'hold': '保持',
            'tap': '轻拍',
            'hammer': '锤击',
            'finger_press': '按压'
        }
        
        # Current labels - always track both
        self.current_object = 'empty'
        self.current_action = 'none'
        
        # Initialize reference images - DISABLED (images not displayed anymore)
        # self._load_reference_images()
    
    def _load_reference_images(self):
        """Load reference images from files"""
        self.object_images = {}
        self.action_images = {}
        
        img_dir = 'reference_images'
        
        print("加载参考图片...")
        
        # Load object images
        for label in self.object_labels:
            img_path = os.path.join(img_dir, 'objects', f'{label}.png')
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    self.object_images[label] = np.array(img)
                    print(f"  加载物体图片: {label}")
                except Exception as e:
                    print(f"  加载失败 {label}: {e}")
                    self.object_images[label] = self._create_fallback_image(label, '#ffe6e6')
            else:
                print(f"  图片不存在，使用占位图: {label}")
                self.object_images[label] = self._create_fallback_image(label, '#ffe6e6')
        
        # Load action images
        for label in self.action_labels:
            img_path = os.path.join(img_dir, 'actions', f'{label}.png')
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    self.action_images[label] = np.array(img)
                    print(f"  加载动作图片: {label}")
                except Exception as e:
                    print(f"  加载失败 {label}: {e}")
                    self.action_images[label] = self._create_fallback_image(label, '#fff8e1')
            else:
                print(f"  图片不存在，使用占位图: {label}")
                self.action_images[label] = self._create_fallback_image(label, '#fff8e1')
        
        print("图片加载完成！\n")
    
    def _create_fallback_image(self, text, bg_color):
        """Create a simple fallback image if file not found"""
        img = Image.new('RGB', (200, 200), bg_color)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((100, 100), text.upper(), fill='#333', anchor='mm', font=font)
        return np.array(img)
        
    def start_collection_gui(self):
        """Start interactive GUI for data collection - Modern Redesigned Interface with Mode Switching"""
        # Setup figure with clean modern style
        plt.style.use('default')
        self.fig = plt.figure(figsize=(18, 11), facecolor='#1a1a2e')
        
        # Title - generic without mode (mode is selected dynamically)
        self.fig.suptitle('Pressure Sensor Data Collection Tool', 
                          fontsize=18, fontweight='bold', color='white', 
                          y=0.98, backgroundcolor='none')
        
        # ==================== MAIN LAYOUT ====================
        # Create grid: More space at bottom for buttons
        # bottom=0.32 ensures graphs stay above y=0.32, leaving space for buttons below
        gs = self.fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35, 
                                   left=0.06, right=0.96, top=0.92, bottom=0.32)
        
        # ==================== PRESSURE HEATMAP (Large, centered) - 高分辨率256x256 ====================
        self.ax_main = self.fig.add_subplot(gs[0:2, 0:2])
        
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
        
        # 使用256x256分辨率显示（通过插值）
        self.im = self.ax_main.imshow(np.zeros((256, 256)), cmap=custom_cmap, 
                                      vmin=0, vmax=255, interpolation='bilinear')
        self.cbar = plt.colorbar(self.im, ax=self.ax_main, fraction=0.046, pad=0.04)
        self.cbar.set_label('Pressure Value', rotation=270, labelpad=22, fontsize=11, color='white')
        self.cbar.ax.tick_params(labelsize=9, colors='white')
        
        # Title with mode indicator
        self.heatmap_title = self.ax_main.set_title('Real-time Pressure Heatmap [DYNAMIC MODE]', 
                                                     fontsize=14, fontweight='bold', pad=15, color='#00FF88')
        self.ax_main.set_xlabel('X Position', fontsize=10, color='white')
        self.ax_main.set_ylabel('Y Position', fontsize=10, color='white')
        self.ax_main.set_facecolor('#0a0a0a')
        self.ax_main.tick_params(colors='white', labelsize=8)
        
        # 添加网格线以显示256个格子
        self.ax_main.set_xticks(np.arange(0, 256, 16))
        self.ax_main.set_yticks(np.arange(0, 256, 16))
        self.ax_main.grid(True, color='#333333', linewidth=0.5, alpha=0.3)
        
        for spine in self.ax_main.spines.values():
            spine.set_edgecolor('#00FF88')
            spine.set_linewidth(2.5)
        
        # Store recent frames for smooth dynamic range
        self.recent_frames = []
        self.max_recent_frames = 10
        
        # ==================== STATISTICS PANEL ====================
        self.ax_stats = self.fig.add_subplot(gs[0, 2])
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor('#16213e')
        rect_stats = plt.Rectangle((0.02, 0.02), 0.96, 0.96, 
                                   transform=self.ax_stats.transAxes,
                                   facecolor='#16213e', edgecolor='#3498db', 
                                   linewidth=3, zorder=0)
        self.ax_stats.add_patch(rect_stats)
        self.stats_text = self.ax_stats.text(
            0.5, 0.5, '', fontsize=13, verticalalignment='center',
            horizontalalignment='center', fontweight='bold', 
            color='#3498db'
        )
        self.ax_stats.text(0.5, 0.88, 'STATISTICS', fontsize=12, 
                          fontweight='bold', color='#3498db',
                          ha='center', va='center',
                          transform=self.ax_stats.transAxes)
        
        # ==================== CURRENT LABEL PANEL ====================
        self.ax_labels = self.fig.add_subplot(gs[1, 2])
        self.ax_labels.axis('off')
        self.ax_labels.set_facecolor('#16213e')
        # Store rect for dynamic color updates
        self.rect_labels = plt.Rectangle((0.02, 0.02), 0.96, 0.96, 
                                         transform=self.ax_labels.transAxes,
                                         facecolor='#16213e', edgecolor='#ff6b6b', 
                                         linewidth=3, zorder=0)
        self.ax_labels.add_patch(self.rect_labels)
        self.labels_text = self.ax_labels.text(
            0.5, 0.5, 'EMPTY', fontsize=24, verticalalignment='center',
            horizontalalignment='center', fontweight='bold',
            family='monospace', color='#ff6b6b', zorder=10
        )
        # Dynamic label title
        self.label_title_text = self.ax_labels.text(0.5, 0.88, 'CURRENT OBJECT', fontsize=12, 
                           fontweight='bold', color='#ff6b6b',
                           ha='center', va='center',
                           transform=self.ax_labels.transAxes)
        
        # ==================== HISTOGRAM ====================
        self.ax_hist = self.fig.add_subplot(gs[2, :])
        self.ax_hist.set_facecolor('#0f1419')
        self.ax_hist.set_title('PRESSURE DISTRIBUTION', fontsize=11, 
                              fontweight='bold', color='#3498db', pad=10)
        self.ax_hist.tick_params(labelsize=8, colors='white')
        self.ax_hist.grid(True, alpha=0.2, color='#3498db', linestyle='--')
        for spine in self.ax_hist.spines.values():
            spine.set_edgecolor('#3498db')
            spine.set_linewidth(1.5)
        
        # ==================== BUTTON AREA (Completely separated) ====================
        # Buttons must be BELOW the gridspec bottom (0.32)
        button_area_y = 0.28  # Top of button area (below gridspec)
        
        # Add visual separator line
        separator_line = plt.Line2D([0.03, 0.97], [button_area_y + 0.01, button_area_y + 0.01], 
                                    transform=self.fig.transFigure, 
                                    color='#3498db', linewidth=2, linestyle='--', alpha=0.6)
        self.fig.add_artist(separator_line)
        
        # Button dimensions
        button_height = 0.055
        button_width = 0.135
        gap_x = 0.012
        
        # ==================== VISUALIZATION MODE BUTTONS (Top row at Y=0.30) ====================
        viz_button_y = 0.30
        viz_button_width = 0.11
        viz_button_height = 0.040
        viz_gap = 0.008
        
        # Add visualization mode label
        self.fig.text(0.06, viz_button_y + viz_button_height + 0.01, 
                     'HEATMAP MODE:', fontsize=11, fontweight='bold', 
                     color='#00FF88', ha='left', va='bottom')
        
        # Absolute Mode Button
        ax_viz_abs = plt.axes([0.20, viz_button_y, viz_button_width, viz_button_height])
        self.btn_viz_abs = Button(ax_viz_abs, 'Absolute', 
                                  color='#34495e', hovercolor='#4a5f7f')
        self.btn_viz_abs.label.set_fontsize(9)
        self.btn_viz_abs.label.set_fontweight('bold')
        self.btn_viz_abs.label.set_color('white')
        self.btn_viz_abs.on_clicked(lambda event: self.switch_viz_mode('absolute'))
        
        # Dynamic Mode Button (default selected)
        ax_viz_dyn = plt.axes([0.20 + viz_button_width + viz_gap, viz_button_y, viz_button_width, viz_button_height])
        self.btn_viz_dyn = Button(ax_viz_dyn, 'Dynamic', 
                                  color='#27ae60', hovercolor='#229954')
        self.btn_viz_dyn.label.set_fontsize(9)
        self.btn_viz_dyn.label.set_fontweight('bold')
        self.btn_viz_dyn.label.set_color('white')
        self.btn_viz_dyn.on_clicked(lambda event: self.switch_viz_mode('dynamic'))
        
        # Difference Mode Button
        ax_viz_diff = plt.axes([0.20 + 2*(viz_button_width + viz_gap), viz_button_y, viz_button_width, viz_button_height])
        self.btn_viz_diff = Button(ax_viz_diff, 'Difference', 
                                   color='#34495e', hovercolor='#4a5f7f')
        self.btn_viz_diff.label.set_fontsize(9)
        self.btn_viz_diff.label.set_fontweight('bold')
        self.btn_viz_diff.label.set_color('white')
        self.btn_viz_diff.on_clicked(lambda event: self.switch_viz_mode('difference'))
        
        # Threshold Mode Button
        ax_viz_thresh = plt.axes([0.20 + 3*(viz_button_width + viz_gap), viz_button_y, viz_button_width, viz_button_height])
        self.btn_viz_thresh = Button(ax_viz_thresh, 'Threshold', 
                                     color='#34495e', hovercolor='#4a5f7f')
        self.btn_viz_thresh.label.set_fontsize(9)
        self.btn_viz_thresh.label.set_fontweight('bold')
        self.btn_viz_thresh.label.set_color('white')
        self.btn_viz_thresh.on_clicked(lambda event: self.switch_viz_mode('threshold'))
        
        # Set Reference Button (for difference mode)
        ax_set_ref = plt.axes([0.20 + 4*(viz_button_width + viz_gap), viz_button_y, viz_button_width, viz_button_height])
        self.btn_set_ref = Button(ax_set_ref, 'Set Ref', 
                                  color='#8e44ad', hovercolor='#7d3c98')
        self.btn_set_ref.label.set_fontsize(9)
        self.btn_set_ref.label.set_fontweight('bold')
        self.btn_set_ref.label.set_color('white')
        self.btn_set_ref.on_clicked(lambda event: self.set_reference_frame())
        
        # ==================== MODE SELECTION BUTTONS (at Y=0.26) ====================
        mode_button_y = 0.26
        mode_button_width = 0.15
        mode_button_height = 0.045
        
        # Add mode selection label
        self.fig.text(0.06, mode_button_y + mode_button_height + 0.01, 
                     'SELECT MODE:', fontsize=13, fontweight='bold', 
                     color='#00d4ff', ha='left', va='bottom')
        
        # Object Mode Button
        ax_mode_object = plt.axes([0.22, mode_button_y, mode_button_width, mode_button_height])
        self.btn_mode_object = Button(ax_mode_object, 'OBJECT', 
                                      color='#e74c3c', hovercolor='#c0392b')
        self.btn_mode_object.label.set_fontsize(11)
        self.btn_mode_object.label.set_fontweight('bold')
        self.btn_mode_object.label.set_color('white')
        self.btn_mode_object.on_clicked(lambda event: self.switch_mode('object'))
        
        # Action Mode Button
        ax_mode_action = plt.axes([0.39, mode_button_y, mode_button_width, mode_button_height])
        self.btn_mode_action = Button(ax_mode_action, 'ACTION', 
                                      color='#34495e', hovercolor='#4a5f7f')
        self.btn_mode_action.label.set_fontsize(11)
        self.btn_mode_action.label.set_fontweight('bold')
        self.btn_mode_action.label.set_color('white')
        self.btn_mode_action.on_clicked(lambda event: self.switch_mode('action'))
        
        # ==================== LABEL SELECTION BUTTONS ====================
        label_buttons_y = 0.19  # Position for label buttons (well below gridspec)
        
        # Section title - will be updated dynamically
        self.section_title_text = self.fig.text(0.06, label_buttons_y + button_height + 0.015, 
                                                'SELECT OBJECT:', fontsize=13, fontweight='bold', 
                                                color='#ff6b6b', ha='left', va='bottom')
        
        # Store button axes for dynamic recreation
        self.label_button_axes = []
        self.label_buttons = []
        
        # Create initial buttons based on current mode
        self._create_label_buttons(label_buttons_y, button_width, button_height, gap_x)
        
        # ==================== ACTION BUTTONS (Capture/Save) ====================
        action_buttons_y = 0.10  # Position for action buttons (below label buttons)
        
        # Capture button (larger and more prominent)
        ax_capture = plt.axes([0.30, action_buttons_y, 0.20, button_height + 0.015])
        self.btn_capture = Button(ax_capture, 'CAPTURE DATA (Press C)', 
                                  color='#27ae60', hovercolor='#229954')
        self.btn_capture.label.set_fontsize(12)
        self.btn_capture.label.set_fontweight('bold')
        self.btn_capture.label.set_color('white')
        self.btn_capture.on_clicked(self.capture_frame)
        
        # Save button
        ax_save = plt.axes([0.53, action_buttons_y, 0.20, button_height + 0.015])
        self.btn_save = Button(ax_save, 'SAVE DATASET (Press S)', 
                              color='#2980b9', hovercolor='#1f5f8b')
        self.btn_save.label.set_fontsize(12)
        self.btn_save.label.set_fontweight('bold')
        self.btn_save.label.set_color('white')
        self.btn_save.on_clicked(self.save_dataset)
        
        # Add keyboard shortcut hint
        hint_text = self.fig.text(0.5, 0.02, 
                                 'Shortcuts: [C] Capture  |  [S] Save  |  [Q] Quit  |  [1-4] Viz Mode  |  [R] Set Reference', 
                                 fontsize=10, color='#95a5a6', ha='center', 
                                 style='italic', weight='bold')
        
        # ==================== EVENT HANDLERS ====================
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Initialize
        self.current_frame = np.zeros((16, 16))
        self.hist_bars = None
        
        # Set initial mode button colors
        if self.mode == 'object':
            self.btn_mode_object.color = '#e74c3c'  # Selected
            self.btn_mode_action.color = '#34495e'  # Unselected
        else:
            self.btn_mode_action.color = '#f39c12'  # Selected
            self.btn_mode_object.color = '#34495e'  # Unselected
        
        # Set initial label display text (use English for better compatibility)
        if self.mode == 'object':
            display_label = self.current_object.upper()
        else:
            display_label = self.current_action.upper()
        self.labels_text.set_text(display_label)
        
        # Start data acquisition
        self.receiver.start_continuous_reading()
        
        # Animation for live updates
        self.ani = FuncAnimation(
            self.fig, 
            self.update_frame, 
            interval=100,
            blit=False,
            cache_frame_data=False
        )
        
        print("GUI启动成功！界面已打开。")
        print(f"初始模式: {self.mode.upper()} Recognition")
        print(f"可视化模式: {self.viz_mode.upper()}")
        print("\n提示:")
        print("  - 点击 OBJECT 或 ACTION 按钮可切换识别模式")
        print("  - 点击可视化模式按钮或按 [1-4] 切换热力图显示方式")
        print("    [1] Absolute    - 固定范围 0-255")
        print("    [2] Dynamic     - 动态自适应范围（推荐）")
        print("    [3] Difference  - 与参考帧的差异")
        print("    [4] Threshold   - 只显示超过阈值的区域")
        print("  - 按 [R] 键更新参考帧（用于差异模式）")
        print("=" * 60)
        
        # Show window
        plt.show()
    
    def update_frame(self, frame_num):
        """Update function called by FuncAnimation"""
        frame_data = self.receiver.get_latest_frame(timeout=0.1)
        
        if frame_data:
            frame = frame_data['frame']  # 16x16原始数据
            
            # Store frame for temporal smoothing in dynamic mode
            self.recent_frames.append(frame.copy())
            if len(self.recent_frames) > self.max_recent_frames:
                self.recent_frames.pop(0)
            
            # Process frame based on visualization mode
            processed_frame, vmin, vmax = self._process_frame_for_display(frame)
            
            # 高质量插值到256x256以显示更多细节
            frame_display = self.receiver.upscale_frame(processed_frame, target_size=256)
            
            # Update main display with processed frame
            self.im.set_data(frame_display)
            
            # Update colorbar limits dynamically
            self.im.set_clim(vmin=vmin, vmax=vmax)
            
            # Update statistics - only every 2 frames to reduce overhead
            if frame_num % 2 == 0:
                stats = self.receiver.get_statistics()
                
                stats_str = (
                    f"FPS: {stats['fps']:.1f}\n"
                    f"Frames: {stats['frames_received']}\n"
                    f"Errors: {stats['errors_count']}\n"
                    f"Samples: {len(self.collected_data)}\n"
                    f"Range: {vmin:.0f}-{vmax:.0f}"
                )
                self.stats_text.set_text(stats_str)
            
            # Update labels display - only every 2 frames (display English)
            if frame_num % 2 == 0:
                if self.mode == 'object':
                    labels_str = self.current_object.upper()
                else:
                    labels_str = self.current_action.upper()
                self.labels_text.set_text(labels_str)
            
            # Update histogram less frequently (every 10 frames) for better performance
            if frame_num % 10 == 0:
                self.ax_hist.clear()
                # Histogram with modern dark theme styling
                self.ax_hist.hist(processed_frame.flatten(), bins=25, color='#3498db', 
                                 alpha=0.8, edgecolor='#2980b9', linewidth=1.2)
                self.ax_hist.set_title('PRESSURE DISTRIBUTION', fontsize=11, 
                                      fontweight='bold', color='#3498db', pad=10)
                self.ax_hist.set_xlabel('Pressure Value', fontsize=9, color='white')
                self.ax_hist.set_ylabel('Frequency', fontsize=9, color='white')
                self.ax_hist.set_facecolor('#0f1419')
                self.ax_hist.tick_params(labelsize=8, colors='white')
                self.ax_hist.grid(True, alpha=0.2, color='#3498db', linestyle='--')
                for spine in self.ax_hist.spines.values():
                    spine.set_edgecolor('#3498db')
                    spine.set_linewidth(1.5)
            
            # Store current frame for capture
            self.current_frame = frame
    
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
                print("Auto-set reference frame")
            
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
        
        # Update title
        mode_names = {
            'absolute': 'ABSOLUTE (0-255)',
            'dynamic': 'DYNAMIC (Auto-Range)',
            'difference': 'DIFFERENCE (vs Reference)',
            'threshold': f'THRESHOLD (>{self.display_threshold})'
        }
        self.heatmap_title.set_text(f'Real-time Pressure Heatmap [{mode_names[new_mode]}]')
        
        # Auto-set reference for difference mode
        if new_mode == 'difference' and self.reference_frame is None and hasattr(self, 'current_frame'):
            self.reference_frame = self.current_frame.copy()
            print("Auto-set reference frame for difference mode")
        
        self.fig.canvas.draw_idle()
        print(f"Switched to {new_mode.upper()} visualization mode")
    
    def set_reference_frame(self):
        """Set current frame as reference for difference mode"""
        if hasattr(self, 'current_frame'):
            self.reference_frame = self.current_frame.copy()
            print("Reference frame updated!")
            print(f"  Reference stats - Min: {np.min(self.reference_frame):.1f}, "
                  f"Max: {np.max(self.reference_frame):.1f}, "
                  f"Mean: {np.mean(self.reference_frame):.1f}")
        else:
            print("No frame available to set as reference")
    
    def _create_label_buttons(self, y_pos, width, height, gap):
        """Create label selection buttons based on current mode"""
        labels = self.object_labels if self.mode == 'object' else self.action_labels
        
        if self.mode == 'object':
            btn_color = '#2c3e50'
            hover_color = '#34495e'
        else:
            btn_color = '#34495e'
            hover_color = '#4a5f7f'
        
        for i, label in enumerate(labels):
            x_pos = 0.06 + i * (width + gap)
            ax_btn = plt.axes([x_pos, y_pos, width, height])
            
            btn = Button(ax_btn, label.upper(), color=btn_color, hovercolor=hover_color)
            btn.label.set_fontsize(10)
            btn.label.set_fontweight('bold')
            btn.label.set_color('white')
            btn.on_clicked(lambda event, lbl=label: self.set_label(lbl))
            
            self.label_buttons.append(btn)
            self.label_button_axes.append(ax_btn)
        
        # Update button colors for default selection
        self._update_button_colors()
    
    def switch_mode(self, new_mode):
        """Switch between object and action recognition modes"""
        if self.mode == new_mode:
            return  # Already in this mode
        
        self.mode = new_mode
        print(f"\n{'='*60}")
        print(f"Switching to {new_mode.upper()} Recognition Mode")
        print(f"{'='*60}")
        
        # Update mode buttons colors
        if new_mode == 'object':
            self.btn_mode_object.color = '#e74c3c'  # Selected - red
            self.btn_mode_object.hovercolor = '#c0392b'
            self.btn_mode_action.color = '#34495e'  # Unselected - gray
            self.btn_mode_action.hovercolor = '#4a5f7f'
            
            # Update section title
            self.section_title_text.set_text('SELECT OBJECT:')
            self.section_title_text.set_color('#ff6b6b')
            
            # Update label panel
            self.label_title_text.set_text('CURRENT OBJECT')
            self.label_title_text.set_color('#ff6b6b')
            self.labels_text.set_color('#ff6b6b')
            self.rect_labels.set_edgecolor('#ff6b6b')
            
        else:  # action mode
            self.btn_mode_action.color = '#f39c12'  # Selected - orange
            self.btn_mode_action.hovercolor = '#e67e22'
            self.btn_mode_object.color = '#34495e'  # Unselected - gray
            self.btn_mode_object.hovercolor = '#4a5f7f'
            
            # Update section title
            self.section_title_text.set_text('SELECT ACTION:')
            self.section_title_text.set_color('#ffa500')
            
            # Update label panel
            self.label_title_text.set_text('CURRENT ACTION')
            self.label_title_text.set_color('#ffa500')
            self.labels_text.set_color('#ffa500')
            self.rect_labels.set_edgecolor('#ffa500')
        
        # Remove old label buttons
        for btn in self.label_buttons:
            btn.ax.remove()
        self.label_buttons.clear()
        self.label_button_axes.clear()
        
        # Recreate label buttons with new labels
        label_buttons_y = 0.19
        button_width = 0.135
        button_height = 0.055
        gap_x = 0.012
        self._create_label_buttons(label_buttons_y, button_width, button_height, gap_x)
        
        # Immediately update the display text with the current label (English)
        if new_mode == 'object':
            display_label = self.current_object.upper()
        else:
            display_label = self.current_action.upper()
        self.labels_text.set_text(display_label)
        
        # Redraw the canvas
        self.fig.canvas.draw_idle()
        
        print(f"Mode switched successfully!")
        if new_mode == 'object':
            print(f"Labels: {', '.join([l.upper() for l in self.object_labels])}")
            print(f"Current: {display_label}")
        else:
            print(f"Labels: {', '.join([l.upper() for l in self.action_labels])}")
            print(f"Current: {display_label}")
        print(f"{'='*60}\n")
    
    def on_close(self, event):
        """Handle window close event"""
        print("Closing application...")
        self.receiver.stop_continuous_reading()
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
    
    def _update_button_colors(self):
        """Update all button colors based on current selection"""
        if self.mode == 'object':
            # Update object buttons
            for i, btn in enumerate(self.label_buttons):
                if self.object_labels[i] == self.current_object:
                    btn.color = '#e74c3c'  # Selected - bright red
                    btn.hovercolor = '#c0392b'
                else:
                    btn.color = '#2c3e50'  # Default - dark blue
                    btn.hovercolor = '#34495e'
        else:  # action mode
            # Update action buttons
            for i, btn in enumerate(self.label_buttons):
                if self.action_labels[i] == self.current_action:
                    btn.color = '#f39c12'  # Selected - bright orange
                    btn.hovercolor = '#e67e22'
                else:
                    btn.color = '#34495e'  # Default - dark gray-blue
                    btn.hovercolor = '#4a5f7f'
    
    def set_label(self, label):
        """Set current label with visual feedback"""
        if self.mode == 'object':
            if self.current_object == label:
                return  # Skip if already selected
            self.current_object = label
            self._update_button_colors()
            # Immediately update the display text with English label
            display_label = label.upper()
            self.labels_text.set_text(display_label)
            self.fig.canvas.draw_idle()  # Refresh button display
            print(f"Object selected: {display_label}")
        else:  # action mode
            if self.current_action == label:
                return  # Skip if already selected
            self.current_action = label
            self._update_button_colors()
            # Immediately update the display text with English label
            display_label = label.upper()
            self.labels_text.set_text(display_label)
            self.fig.canvas.draw_idle()  # Refresh button display
            print(f"Action selected: {display_label}")
    
    def capture_frame(self, event=None):
        """Capture current frame with labels"""
        if hasattr(self, 'current_frame'):
            data_point = {
                'frame': self.current_frame.copy(),
                'object': self.current_object,
                'action': self.current_action,
                'timestamp': time.time(),
                'mode': self.mode  # Record which mode was active
            }
            self.collected_data.append(data_point)
            
            if self.mode == 'object':
                cn_label = self.object_labels_cn.get(self.current_object, self.current_object)
                print(f"已采集 #{len(self.collected_data)}: 物体 = {cn_label} ({self.current_object})")
            else:
                cn_label = self.action_labels_cn.get(self.current_action, self.current_action)
                print(f"已采集 #{len(self.collected_data)}: 动作 = {cn_label} ({self.current_action})")
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'c':
            self.capture_frame()
        elif event.key == 's':
            self.save_dataset()
        elif event.key == 'q':
            plt.close(self.fig)
        # Visualization mode shortcuts
        elif event.key == '1':
            self.switch_viz_mode('absolute')
        elif event.key == '2':
            self.switch_viz_mode('dynamic')
        elif event.key == '3':
            self.switch_viz_mode('difference')
        elif event.key == '4':
            self.switch_viz_mode('threshold')
        elif event.key == 'r':
            self.set_reference_frame()
    
    def save_dataset(self, event=None):
        """Save collected dataset to HDF5 file"""
        if len(self.collected_data) == 0:
            print("No data to save!")
            return
        
        # Generate filename with timestamp and mode
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'pressure_{self.mode}_dataset_{timestamp}.h5'
        filepath = os.path.join(self.save_dir, filename)
        
        # Save to HDF5
        with h5py.File(filepath, 'w') as f:
            # Create datasets
            n_samples = len(self.collected_data)
            
            frames = np.array([d['frame'] for d in self.collected_data])
            objects = np.array([d['object'] for d in self.collected_data], dtype='S20')
            actions = np.array([d['action'] for d in self.collected_data], dtype='S20')
            timestamps = np.array([d['timestamp'] for d in self.collected_data])
            
            f.create_dataset('frames', data=frames)
            f.create_dataset('objects', data=objects)
            f.create_dataset('actions', data=actions)
            f.create_dataset('timestamps', data=timestamps)
            
            # Save metadata
            f.attrs['n_samples'] = n_samples
            f.attrs['frame_shape'] = (16, 16)
            f.attrs['object_labels'] = json.dumps(self.object_labels)
            f.attrs['action_labels'] = json.dumps(self.action_labels)
        
        print(f"SAVED: {n_samples} samples -> {filepath}")
        
        # Clear collected data
        self.collected_data = []
        print("Cleared collected data buffer")


if __name__ == '__main__':
    import sys
    
    # Determine mode from command line argument
    mode = 'object'  # Default mode
    if len(sys.argv) > 1:
        if sys.argv[1] in ['object', 'action']:
            mode = sys.argv[1]
        else:
            print("Usage: python data_collector.py [object|action]")
            print("  object - Collect object recognition data (default)")
            print("  action - Collect action recognition data")
            sys.exit(1)
    
    print("=" * 60)
    print(f"Data Collection Mode: {mode.upper()}")
    print("=" * 60)
    if mode == 'object':
        print("Collecting OBJECT recognition data")
        print("   Objects: ball, bottle, empty, pen, phone, hand")
        print("   (Action will be fixed as 'none')")
    else:
        print("Collecting ACTION recognition data")
        print("   Actions: none, press, rotate, tap, touch, hold")
        print("   (Object will be fixed as 'empty')")
    print("=" * 60)
    print()
    
    # Create receiver with strong noise reduction (针对高帧率优化)
    receiver = PressureSensorReceiver(
        port='COM3',  # Change to your port
        enable_noise_reduction=True,
        noise_threshold=15,  # 适中阈值
        temporal_smoothing=0.7,  # 增强时域平滑（从0.6提高到0.7）
        spatial_smoothing=0.8,  # 增强空间平滑（从0.7提高到0.8）
        multi_frame_average=True,  # 启用多帧平均
        average_frames=10  # 增加到10帧（从6提高到10），大幅降低噪声
    )
    
    # Connect
    if not receiver.connect():
        print("Failed to connect. Using simulation mode...")
        # You can add simulation mode here for testing
    else:
        # 连接成功后进行背景校准
        print("\n" + "="*60)
        print("数据采集前需要先进行背景校准")
        print("="*60)
        print("重要提示:")
        print("   1. 请确保传感器表面干净,没有任何物体")
        print("   2. 确保传感器处于稳定状态")
        print("   3. 背景校准需要约1秒钟")
        print("="*60)
        input("\n准备好后按回车键开始背景校准...")
        
        receiver.calibrate_background(num_samples=20)
        
        print("\n背景校准完成! 可以开始数据采集了")
        print("="*60 + "\n")
    
    # Create data collector with specified mode
    collector = DataCollector(receiver, mode=mode)
    
    # Start GUI
    collector.start_collection_gui()
