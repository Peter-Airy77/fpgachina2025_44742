"""
机械臂触碰报警系统 (Robotic Arm Touch Alert System)
通过压力传感器实时监测机械臂的触碰，当检测到触碰时立即报警

主要功能：
1. 实时监控压力传感器数据
2. 智能触碰检测（基于阈值和变化率）
3. 多种报警方式：声音、视觉闪烁、控制台提示
4. 可调灵敏度和报警参数
5. 触碰事件记录和统计
6. 实时可视化界面
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import time
from datetime import datetime
from collections import deque
import winsound  # Windows系统声音报警
import threading
import json
import os
import warnings

# 配置matplotlib字体，避免中文显示问题
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 如果中文字体不可用，使用英文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# 导入串口接收器
from serial_receiver_强力降噪版 import PressureSensorReceiver


class TouchAlertSystem:
    """
    机械臂触碰报警系统
    """
    
    def __init__(self, receiver, config=None):
        """
        初始化触碰报警系统
        
        Args:
            receiver: PressureSensorReceiver实例
            config: 配置字典（可选）
        """
        self.receiver = receiver
        
        # ========== 触碰检测参数 ==========
        self.config = config or {}
        
        # 压力阈值（超过此值认为有触碰）
        self.pressure_threshold = self.config.get('pressure_threshold', 30)
        
        # 触碰区域最小面积（像素数）
        self.min_touch_area = self.config.get('min_touch_area', 3)
        
        # 压力变化率阈值（用于检测突然的触碰）
        self.change_rate_threshold = self.config.get('change_rate_threshold', 15)
        
        # 报警冷却时间（秒）- 避免频繁报警
        self.alert_cooldown = self.config.get('alert_cooldown', 1.0)
        
        # 报警持续时间（秒）
        self.alert_duration = self.config.get('alert_duration', 2.0)
        
        # ========== 状态变量 ==========
        self.is_alert_active = False  # 当前是否处于报警状态
        self.last_alert_time = 0  # 上次报警时间
        self.alert_start_time = 0  # 当前报警开始时间
        
        # 触碰事件记录
        self.touch_events = []
        self.max_events = 100  # 最多记录100个事件
        
        # 用于检测变化率的历史帧
        self.frame_history = deque(maxlen=5)
        
        # 统计信息
        self.total_touches = 0
        self.false_alarms = 0  # 可以手动标记误报
        
        # 背景参考帧（用于检测变化）
        self.reference_frame = None
        self.auto_update_reference = True  # 是否自动更新参考帧
        self.reference_update_interval = 5.0  # 参考帧更新间隔（秒）
        self.last_reference_update = time.time()
        
        # 报警声音设置
        self.alert_sound_enabled = self.config.get('alert_sound', True)
        self.alert_frequency = 1000  # Hz
        self.alert_sound_duration = 300  # ms
        
        print("\n" + "="*70)
        print("🤖 机械臂触碰报警系统已初始化")
        print("="*70)
        print(f"📊 检测参数:")
        print(f"   压力阈值:       {self.pressure_threshold}")
        print(f"   最小触碰面积:   {self.min_touch_area} 像素")
        print(f"   变化率阈值:     {self.change_rate_threshold}")
        print(f"   报警冷却时间:   {self.alert_cooldown} 秒")
        print(f"   报警持续时间:   {self.alert_duration} 秒")
        print(f"   声音报警:       {'启用' if self.alert_sound_enabled else '禁用'}")
        print("="*70 + "\n")
    
    def detect_touch(self, frame):
        """
        检测触碰事件
        
        Args:
            frame: 压力传感器数据帧 (16x16)
            
        Returns:
            dict: 触碰检测结果
                - is_touched: 是否检测到触碰
                - touch_intensity: 触碰强度（0-100）
                - touch_area: 触碰区域面积（像素数）
                - touch_location: 触碰中心位置 (x, y)
                - pressure_change: 压力变化率
        """
        result = {
            'is_touched': False,
            'touch_intensity': 0.0,
            'touch_area': 0,
            'touch_location': None,
            'pressure_change': 0.0,
            'max_pressure': 0.0
        }
        
        # 添加到历史帧
        self.frame_history.append(frame.copy())
        
        # 计算最大压力
        max_pressure = np.max(frame)
        result['max_pressure'] = float(max_pressure)
        
        # 方法1: 基于绝对阈值检测
        touch_mask = frame > self.pressure_threshold
        touch_area = np.sum(touch_mask)
        result['touch_area'] = int(touch_area)
        
        # 方法2: 基于变化率检测（如果有历史数据）
        pressure_change = 0.0
        if len(self.frame_history) >= 2:
            prev_frame = self.frame_history[-2]
            diff = frame - prev_frame
            pressure_change = np.max(np.abs(diff))
            result['pressure_change'] = float(pressure_change)
        
        # 方法3: 基于背景差分（如果有参考帧）
        background_diff = 0.0
        if self.reference_frame is not None:
            diff_from_ref = frame.astype(np.float32) - self.reference_frame.astype(np.float32)
            diff_from_ref = np.maximum(diff_from_ref, 0)  # 只关心增加的压力
            background_diff = np.max(diff_from_ref)
        
        # 综合判断：任意条件满足即认为有触碰
        is_touched = False
        
        # 条件1: 压力超过阈值且面积足够大
        if touch_area >= self.min_touch_area and max_pressure > self.pressure_threshold:
            is_touched = True
        
        # 条件2: 压力突然变化（快速触碰）
        if pressure_change > self.change_rate_threshold:
            is_touched = True
        
        # 条件3: 相对于背景有显著变化
        if self.reference_frame is not None and background_diff > self.pressure_threshold:
            is_touched = True
        
        result['is_touched'] = is_touched
        
        # 计算触碰强度（0-100）
        if is_touched:
            # 基于最大压力计算强度
            intensity = min(100, (max_pressure / 255.0) * 100)
            result['touch_intensity'] = float(intensity)
            
            # 计算触碰中心位置（加权平均）
            if touch_area > 0:
                y_indices, x_indices = np.where(touch_mask)
                weights = frame[touch_mask]
                if np.sum(weights) > 0:
                    center_x = np.average(x_indices, weights=weights)
                    center_y = np.average(y_indices, weights=weights)
                    result['touch_location'] = (float(center_x), float(center_y))
        
        return result
    
    def trigger_alert(self, touch_info):
        """
        触发报警
        
        Args:
            touch_info: 触碰检测结果
        """
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        # 激活报警
        self.is_alert_active = True
        self.alert_start_time = current_time
        self.last_alert_time = current_time
        self.total_touches += 1
        
        # 记录触碰事件
        event = {
            'timestamp': datetime.now(),
            'touch_intensity': touch_info['touch_intensity'],
            'touch_area': touch_info['touch_area'],
            'touch_location': touch_info['touch_location'],
            'max_pressure': touch_info['max_pressure'],
            'pressure_change': touch_info['pressure_change']
        }
        self.touch_events.append(event)
        if len(self.touch_events) > self.max_events:
            self.touch_events.pop(0)
        
        # 控制台报警
        self._console_alert(touch_info)
        
        # 声音报警（在新线程中执行，避免阻塞）
        if self.alert_sound_enabled:
            threading.Thread(target=self._sound_alert, daemon=True).start()
    
    def _console_alert(self, touch_info):
        """控制台文字报警"""
        print("\n" + "="*70)
        print("🚨 【报警】检测到触碰！")
        print("="*70)
        print(f"⏰ 时间:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"💪 触碰强度:   {touch_info['touch_intensity']:.1f}%")
        print(f"📍 触碰面积:   {touch_info['touch_area']} 像素")
        print(f"🎯 最大压力:   {touch_info['max_pressure']:.1f}")
        print(f"⚡ 压力变化:   {touch_info['pressure_change']:.1f}")
        if touch_info['touch_location']:
            x, y = touch_info['touch_location']
            print(f"📌 触碰位置:   ({x:.1f}, {y:.1f})")
        print(f"🔢 累计触碰:   {self.total_touches} 次")
        print("="*70 + "\n")
    
    def _sound_alert(self):
        """声音报警"""
        try:
            # 播放警报声（Windows）
            for _ in range(3):  # 连续响3次
                winsound.Beep(self.alert_frequency, self.alert_sound_duration)
                time.sleep(0.1)
        except Exception as e:
            print(f"声音报警失败: {e}")
    
    def update_alert_state(self):
        """更新报警状态（用于持续报警效果）"""
        if self.is_alert_active:
            elapsed = time.time() - self.alert_start_time
            if elapsed >= self.alert_duration:
                self.is_alert_active = False
    
    def update_reference_frame(self, frame):
        """更新背景参考帧"""
        self.reference_frame = frame.copy()
        self.last_reference_update = time.time()
        print(f"✅ 参考帧已更新 (背景平均值: {frame.mean():.1f})")
    
    def auto_update_reference_check(self, frame):
        """自动更新参考帧检查"""
        if self.auto_update_reference and not self.is_alert_active:
            current_time = time.time()
            if current_time - self.last_reference_update > self.reference_update_interval:
                # 只在没有触碰时更新参考帧
                touch_result = self.detect_touch(frame)
                if not touch_result['is_touched']:
                    self.update_reference_frame(frame)
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'total_touches': self.total_touches,
            'false_alarms': self.false_alarms,
            'events_count': len(self.touch_events),
            'is_alert_active': self.is_alert_active,
            'reference_frame_age': time.time() - self.last_reference_update if self.reference_frame is not None else None
        }
    
    def save_configuration(self, filepath='touch_alert_config.json'):
        """保存配置到文件"""
        config = {
            'pressure_threshold': self.pressure_threshold,
            'min_touch_area': self.min_touch_area,
            'change_rate_threshold': self.change_rate_threshold,
            'alert_cooldown': self.alert_cooldown,
            'alert_duration': self.alert_duration,
            'alert_sound': self.alert_sound_enabled
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"✅ 配置已保存到 {filepath}")
    
    def load_configuration(self, filepath='touch_alert_config.json'):
        """从文件加载配置"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.pressure_threshold = config.get('pressure_threshold', self.pressure_threshold)
            self.min_touch_area = config.get('min_touch_area', self.min_touch_area)
            self.change_rate_threshold = config.get('change_rate_threshold', self.change_rate_threshold)
            self.alert_cooldown = config.get('alert_cooldown', self.alert_cooldown)
            self.alert_duration = config.get('alert_duration', self.alert_duration)
            self.alert_sound_enabled = config.get('alert_sound', self.alert_sound_enabled)
            
            print(f"✅ 配置已从 {filepath} 加载")
        else:
            print(f"⚠️  配置文件 {filepath} 不存在")
    
    def start_monitoring(self):
        """启动监控和可视化界面"""
        # 配置matplotlib样式
        plt.style.use('dark_background')
        
        # 创建窗口
        self.fig = plt.figure(figsize=(18, 10), facecolor='#0a0e27')
        self.fig.suptitle('Robotic Arm Touch Alert System - Real-time Monitoring', 
                         fontsize=18, fontweight='bold', color='#00d9ff', y=0.98)
        
        # 创建布局
        gs = self.fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30,
                                   left=0.05, right=0.97, top=0.94, bottom=0.06)
        
        # ===== 左上：压力热力图 =====
        self.ax_heatmap = self.fig.add_subplot(gs[0:2, 0:2])
        
        # 使用高对比度的colormap
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = [
            (0.0, '#000000'),   # 黑色
            (0.15, '#0d47a1'),  # 深蓝
            (0.30, '#1976d2'),  # 蓝色
            (0.45, '#fbc02d'),  # 黄色
            (0.60, '#ff9800'),  # 橙色
            (0.75, '#f44336'),  # 红色
            (1.0, '#ffffff')    # 白色（警告）
        ]
        positions = [c[0] for c in colors_list]
        colors = [c[1] for c in colors_list]
        custom_cmap = LinearSegmentedColormap.from_list('alert_cmap',
                                                        list(zip(positions, colors)), N=256)
        
        self.im_heatmap = self.ax_heatmap.imshow(np.zeros((256, 256)),
                                                  cmap=custom_cmap,
                                                  vmin=0, vmax=255,
                                                  interpolation='bilinear')
        
        self.ax_heatmap.set_title('Pressure Distribution Heatmap', fontsize=14,
                                 fontweight='bold', color='#00d9ff', pad=10)
        self.ax_heatmap.set_xlabel('X Coordinate', fontsize=10, color='white')
        self.ax_heatmap.set_ylabel('Y Coordinate', fontsize=10, color='white')
        self.ax_heatmap.tick_params(colors='white', labelsize=8)
        
        # 添加网格
        self.ax_heatmap.set_xticks(np.arange(0, 256, 32))
        self.ax_heatmap.set_yticks(np.arange(0, 256, 32))
        self.ax_heatmap.grid(True, color='#333333', linewidth=0.5, alpha=0.3)
        
        # 添加颜色条
        self.cbar = plt.colorbar(self.im_heatmap, ax=self.ax_heatmap,
                                fraction=0.046, pad=0.04)
        self.cbar.set_label('压力值', rotation=270, labelpad=20,
                           fontsize=10, color='white')
        self.cbar.ax.tick_params(colors='white', labelsize=8)
        
        for spine in self.ax_heatmap.spines.values():
            spine.set_edgecolor('#00d9ff')
            spine.set_linewidth(2.5)
        
        # ===== 右上：报警状态显示 =====
        self.ax_alert = self.fig.add_subplot(gs[0, 2])
        self.ax_alert.set_xlim(0, 1)
        self.ax_alert.set_ylim(0, 1)
        self.ax_alert.axis('off')
        self.ax_alert.set_facecolor('#0f1535')
        
        # 报警指示灯（圆形）
        self.alert_indicator = Circle((0.5, 0.65), 0.20,
                                     facecolor='#2e7d32', edgecolor='white',
                                     linewidth=3, zorder=10)
        self.ax_alert.add_patch(self.alert_indicator)
        
        # 报警文本
        self.alert_text = self.ax_alert.text(0.5, 0.3, 'NORMAL',
                                             fontsize=18, fontweight='bold',
                                             ha='center', va='center',
                                             color='#4caf50')
        
        # 标题
        self.ax_alert.text(0.5, 0.95, 'Alert Status',
                          fontsize=12, fontweight='bold',
                          ha='center', va='center', color='#00d9ff')
        
        # ===== 右中：统计信息 =====
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor('#0f1535')
        
        self.ax_stats.text(0.5, 0.95, 'Statistics',
                          fontsize=12, fontweight='bold',
                          ha='center', va='center', color='#00d9ff')
        
        self.stats_text = self.ax_stats.text(0.5, 0.45, '',
                                            fontsize=10, ha='center', va='center',
                                            color='white', family='monospace',
                                            linespacing=1.8)
        
        # ===== 下方：触碰事件历史 =====
        self.ax_events = self.fig.add_subplot(gs[2, :])
        self.ax_events.set_facecolor('#0f1419')
        self.ax_events.set_title('Touch Event History', fontsize=12,
                                fontweight='bold', color='#00d9ff', pad=10)
        self.ax_events.set_xlabel('Time (seconds)', fontsize=10, color='white')
        self.ax_events.set_ylabel('Touch Intensity (%)', fontsize=10, color='white')
        self.ax_events.tick_params(colors='white', labelsize=8)
        self.ax_events.grid(True, alpha=0.3, color='#3498db', linestyle='--')
        
        for spine in self.ax_events.spines.values():
            spine.set_edgecolor('#3498db')
            spine.set_linewidth(1.5)
        
        # 触碰历史数据
        self.touch_history_times = []
        self.touch_history_intensities = []
        self.max_history_points = 100
        
        # 启动数据接收
        self.receiver.start_continuous_reading()
        
        # 键盘事件处理
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # 启动动画
        self.ani = FuncAnimation(
            self.fig,
            self.update_visualization,
            interval=100,  # 10 FPS
            blit=False,
            cache_frame_data=False
        )
        
        print("\n✅ 监控系统已启动！")
        print("\n快捷键:")
        print("  [R] - 手动更新参考帧")
        print("  [S] - 保存当前配置")
        print("  [+] - 增加灵敏度（降低阈值）")
        print("  [-] - 降低灵敏度（提高阈值）")
        print("  [M] - 静音/取消静音")
        print("  [Q] - 退出程序")
        print("\n开始监控...\n")
        
        plt.show()
    
    def update_visualization(self, frame_num):
        """更新可视化（动画回调）"""
        # 获取最新帧
        frame_data = self.receiver.get_latest_frame(timeout=0.1)
        
        if frame_data:
            frame = frame_data['frame']
            
            # 自动更新参考帧检查
            self.auto_update_reference_check(frame)
            
            # 检测触碰
            touch_result = self.detect_touch(frame)
            
            # 如果检测到触碰且不在冷却期，触发报警
            if touch_result['is_touched']:
                self.trigger_alert(touch_result)
            
            # 更新报警状态
            self.update_alert_state()
            
            # 更新热力图
            frame_display = self.receiver.upscale_frame(frame, target_size=256)
            self.im_heatmap.set_data(frame_display)
            
            # 动态调整颜色范围
            vmin, vmax = np.percentile(frame_display, [2, 98])
            if vmax - vmin < 10:
                vmin, vmax = 0, max(10, vmax)
            self.im_heatmap.set_clim(vmin=vmin, vmax=vmax)
            
            # 更新报警指示器
            if self.is_alert_active:
                # 闪烁效果
                flash = (time.time() * 5) % 1 < 0.5
                if flash:
                    self.alert_indicator.set_facecolor('#f44336')  # 红色
                    self.alert_text.set_text('! TOUCH !')
                    self.alert_text.set_color('#f44336')
                else:
                    self.alert_indicator.set_facecolor('#ff9800')  # 橙色
                    self.alert_text.set_text('! ALERT !')
                    self.alert_text.set_color('#ff9800')
            else:
                self.alert_indicator.set_facecolor('#2e7d32')  # 绿色
                self.alert_text.set_text('NORMAL')
                self.alert_text.set_color('#4caf50')
            
            # 更新统计信息
            stats = self.get_statistics()
            receiver_stats = self.receiver.get_statistics()
            
            stats_str = (
                f"Touches:        {stats['total_touches']:3d}\n"
                f"Intensity:      {touch_result['touch_intensity']:5.1f}%\n"
                f"Touch Area:     {touch_result['touch_area']:3d} px\n"
                f"Max Pressure:   {touch_result['max_pressure']:5.1f}\n"
                f"Change Rate:    {touch_result['pressure_change']:5.1f}\n"
                f"\n"
                f"FPS:            {receiver_stats['fps']:5.1f}\n"
                f"Frames:         {receiver_stats['frames_received']:5d}\n"
                f"\n"
                f"Sensitivity:    {self.pressure_threshold:3d}"
            )
            self.stats_text.set_text(stats_str)
            
            # 更新触碰历史曲线
            if touch_result['is_touched']:
                current_time = time.time()
                if len(self.touch_history_times) == 0:
                    start_time = current_time
                else:
                    start_time = self.touch_history_times[0][1]
                
                self.touch_history_times.append((current_time - start_time, current_time))
                self.touch_history_intensities.append(touch_result['touch_intensity'])
                
                # 限制历史点数
                if len(self.touch_history_times) > self.max_history_points:
                    self.touch_history_times.pop(0)
                    self.touch_history_intensities.pop(0)
            
            # 绘制触碰历史
            if len(self.touch_history_times) > 0:
                self.ax_events.clear()
                
                times = [t[0] for t in self.touch_history_times]
                intensities = self.touch_history_intensities
                
                # 绘制曲线和散点
                self.ax_events.plot(times, intensities,
                                   color='#ff6b35', linewidth=2, alpha=0.7)
                self.ax_events.scatter(times, intensities,
                                      color='#f44336', s=50, alpha=0.8, zorder=10)
                
                # 添加阈值线
                if len(times) > 0:
                    self.ax_events.axhline(y=50, color='#fbc02d',
                                          linestyle='--', linewidth=1.5,
                                          alpha=0.6, label='Medium')
                    self.ax_events.axhline(y=80, color='#f44336',
                                          linestyle='--', linewidth=1.5,
                                          alpha=0.6, label='High')
                
                self.ax_events.set_title('Touch Event History', fontsize=12,
                                        fontweight='bold', color='#00d9ff', pad=10)
                self.ax_events.set_xlabel('Time (seconds)', fontsize=10, color='white')
                self.ax_events.set_ylabel('Touch Intensity (%)', fontsize=10, color='white')
                self.ax_events.set_ylim([0, 105])
                self.ax_events.tick_params(colors='white', labelsize=8)
                self.ax_events.grid(True, alpha=0.3, color='#3498db', linestyle='--')
                self.ax_events.legend(loc='upper left', fontsize=8)
                self.ax_events.set_facecolor('#0f1419')
                
                for spine in self.ax_events.spines.values():
                    spine.set_edgecolor('#3498db')
                    spine.set_linewidth(1.5)
            
            plt.pause(0.001)
    
    def on_key_press(self, event):
        """键盘事件处理"""
        if event.key == 'r':
            # 手动更新参考帧
            if hasattr(self, 'receiver'):
                frame_data = self.receiver.get_latest_frame(timeout=0.1)
                if frame_data:
                    self.update_reference_frame(frame_data['frame'])
        
        elif event.key == 's':
            # 保存配置
            self.save_configuration()
        
        elif event.key == '+' or event.key == '=':
            # 增加灵敏度（降低阈值）
            self.pressure_threshold = max(5, self.pressure_threshold - 5)
            print(f"🔧 灵敏度提高，阈值降低至: {self.pressure_threshold}")
        
        elif event.key == '-' or event.key == '_':
            # 降低灵敏度（提高阈值）
            self.pressure_threshold = min(100, self.pressure_threshold + 5)
            print(f"🔧 灵敏度降低，阈值提高至: {self.pressure_threshold}")
        
        elif event.key == 'm':
            # 切换静音
            self.alert_sound_enabled = not self.alert_sound_enabled
            status = "启用" if self.alert_sound_enabled else "禁用"
            print(f"🔊 声音报警已{status}")
        
        elif event.key == 'q':
            # 退出
            print("\n退出监控系统...")
            plt.close(self.fig)
    
    def on_close(self, event):
        """窗口关闭事件"""
        print("\n关闭监控系统...")
        self.receiver.stop_continuous_reading()
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🤖 机械臂触碰报警系统")
    print("="*70)
    print("\n系统功能：")
    print("  ✓ 实时监控压力传感器数据")
    print("  ✓ 智能触碰检测（多重判断机制）")
    print("  ✓ 声音+视觉双重报警")
    print("  ✓ 触碰事件记录与统计")
    print("  ✓ 可调灵敏度和参数")
    print("  ✓ 实时可视化界面")
    print("\n" + "="*70 + "\n")
    
    # 创建串口接收器（使用强力降噪）
    print("正在初始化串口接收器...")
    receiver = PressureSensorReceiver(
        port='COM3',
        baudrate=115200,
        enable_noise_reduction=True,
        noise_threshold=15,
        temporal_smoothing=0.7,
        spatial_smoothing=0.8,
        multi_frame_average=True,
        average_frames=10,
        display_fps_limit=20
    )
    
    # 连接串口
    if not receiver.connect():
        print("❌ 串口连接失败！")
        print("\n请检查：")
        print("  1. 串口号是否正确（当前: COM3）")
        print("  2. FPGA设备是否已上电")
        print("  3. 是否有其他程序占用串口")
        return
    
    print("✅ 串口连接成功！")
    
    # 背景校准
    print("\n" + "="*70)
    print("📊 背景校准")
    print("="*70)
    print("⚠️  重要提示：")
    print("   1. 请确保机械臂传感器表面无任何接触")
    print("   2. 确保机械臂处于静止状态")
    print("   3. 校准需要约1秒钟")
    print("="*70)
    input("\n准备好后按回车键开始校准...")
    
    receiver.calibrate_background(num_samples=30)
    
    print("\n✅ 背景校准完成！")
    
    # 加载或创建配置
    config_file = 'touch_alert_config.json'
    if os.path.exists(config_file):
        print(f"\n📄 找到配置文件: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ 配置已加载")
    else:
        print(f"\n📄 未找到配置文件，使用默认配置")
        config = {
            'pressure_threshold': 30,
            'min_touch_area': 3,
            'change_rate_threshold': 15,
            'alert_cooldown': 1.0,
            'alert_duration': 2.0,
            'alert_sound': True
        }
    
    # 创建触碰报警系统
    alert_system = TouchAlertSystem(receiver, config=config)
    
    # 设置初始参考帧
    print("\n正在获取初始参考帧...")
    time.sleep(0.5)
    frame_data = receiver.get_latest_frame(timeout=2.0)
    if frame_data:
        alert_system.update_reference_frame(frame_data['frame'])
    
    # 启动监控
    print("\n" + "="*70)
    print("🚀 启动监控系统")
    print("="*70)
    print("\n提示：")
    print("  - 现在可以触碰机械臂测试报警功能")
    print("  - 使用快捷键调整系统参数")
    print("  - 关闭窗口或按 [Q] 退出")
    print("\n" + "="*70 + "\n")
    
    try:
        alert_system.start_monitoring()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断，正在退出...")
    finally:
        receiver.disconnect()
        print("✅ 程序已退出")


if __name__ == '__main__':
    main()

