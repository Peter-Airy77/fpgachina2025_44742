"""
Generate Reference Images for Real-time Inference System
为实时推理系统生成参考图片
"""

import sys
import io

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge, Polygon
import numpy as np
import os
from pathlib import Path

# 设置中文字体
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False


class ReferenceImageGenerator:
    """生成参考图片的类"""
    
    def __init__(self, output_dir='reference_images', img_size=(400, 400), dpi=100):
        self.output_dir = output_dir
        self.img_size = img_size
        self.dpi = dpi
        
        # 创建输出目录
        self.objects_dir = os.path.join(output_dir, 'objects')
        self.actions_dir = os.path.join(output_dir, 'actions')
        Path(self.objects_dir).mkdir(parents=True, exist_ok=True)
        Path(self.actions_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ 创建输出目录:")
        print(f"  - {self.objects_dir}")
        print(f"  - {self.actions_dir}")
    
    def create_base_figure(self, bg_color='#1a1a2e'):
        """创建基础画布"""
        fig, ax = plt.subplots(figsize=(self.img_size[0]/self.dpi, self.img_size[1]/self.dpi), 
                               dpi=self.dpi, facecolor=bg_color)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor(bg_color)
        return fig, ax
    
    def save_and_close(self, fig, filepath, name):
        """保存并关闭图形"""
        plt.tight_layout(pad=0.5)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"  ✓ {name}: {filepath}")
    
    # ========== 物品图片生成 ==========
    
    def generate_empty(self):
        """生成 empty (空) 图片"""
        fig, ax = self.create_base_figure(bg_color='#0a0e27')
        
        # 绘制虚线边框
        border = FancyBboxPatch((1, 1), 8, 8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#4a5568', 
                               facecolor='none',
                               linestyle='--',
                               linewidth=3,
                               alpha=0.5)
        ax.add_patch(border)
        
        # 添加文字
        ax.text(5, 6, 'EMPTY', fontsize=48, fontweight='bold',
               ha='center', va='center', color='#718096', alpha=0.6)
        ax.text(5, 3.5, '无物体', fontsize=32, fontweight='bold',
               ha='center', va='center', color='#4a5568', alpha=0.5)
        
        filepath = os.path.join(self.objects_dir, 'empty.png')
        self.save_and_close(fig, filepath, 'Empty')
    
    def generate_ball(self):
        """生成 ball (球) 图片"""
        fig, ax = self.create_base_figure(bg_color='#1a1a2e')
        
        # 绘制球体（带阴影效果）
        # 阴影
        shadow = Circle((5.2, 4.8), 2.8, color='#000000', alpha=0.3, zorder=1)
        ax.add_patch(shadow)
        
        # 主球体
        ball = Circle((5, 5), 2.5, color='#f59e0b', zorder=2)
        ax.add_patch(ball)
        
        # 高光效果
        highlight = Circle((4.2, 6), 0.8, color='#fef3c7', alpha=0.6, zorder=3)
        ax.add_patch(highlight)
        
        # 纹理线条
        for angle in range(0, 180, 30):
            theta = np.radians(angle)
            x1 = 5 + 2.5 * np.cos(theta)
            y1 = 5 + 2.5 * np.sin(theta)
            x2 = 5 - 2.5 * np.cos(theta)
            y2 = 5 - 2.5 * np.sin(theta)
            ax.plot([x1, x2], [y1, y2], color='#d97706', linewidth=2, alpha=0.4, zorder=2)
        
        # 标签
        ax.text(5, 1.2, 'BALL', fontsize=32, fontweight='bold',
               ha='center', va='center', color='#f59e0b')
        ax.text(5, 0.5, '球', fontsize=24,
               ha='center', va='center', color='#fbbf24')
        
        filepath = os.path.join(self.objects_dir, 'ball.png')
        self.save_and_close(fig, filepath, 'Ball')
    
    def generate_bottle(self):
        """生成 bottle (瓶子) 图片"""
        fig, ax = self.create_base_figure(bg_color='#1a1a2e')
        
        # 瓶身
        bottle_body = Rectangle((3.5, 2), 3, 5, 
                               facecolor='#3b82f6', 
                               edgecolor='#1e40af', 
                               linewidth=3, zorder=2)
        ax.add_patch(bottle_body)
        
        # 瓶颈
        bottle_neck = Rectangle((4.2, 7), 1.6, 1.5,
                               facecolor='#60a5fa',
                               edgecolor='#1e40af',
                               linewidth=3, zorder=2)
        ax.add_patch(bottle_neck)
        
        # 瓶盖
        bottle_cap = Rectangle((4, 8.5), 2, 0.8,
                              facecolor='#ef4444',
                              edgecolor='#991b1b',
                              linewidth=2, zorder=3)
        ax.add_patch(bottle_cap)
        
        # 标签
        label_rect = FancyBboxPatch((3.8, 4), 2.4, 1.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#dbeafe',
                                   edgecolor='#1e40af',
                                   linewidth=2, zorder=3)
        ax.add_patch(label_rect)
        
        # 高光
        highlight = Rectangle((3.6, 3), 0.3, 3.5,
                             facecolor='#93c5fd',
                             alpha=0.4, zorder=3)
        ax.add_patch(highlight)
        
        # 文字标签
        ax.text(5, 1.2, 'BOTTLE', fontsize=32, fontweight='bold',
               ha='center', va='center', color='#3b82f6')
        ax.text(5, 0.5, '瓶子', fontsize=24,
               ha='center', va='center', color='#60a5fa')
        
        filepath = os.path.join(self.objects_dir, 'bottle.png')
        self.save_and_close(fig, filepath, 'Bottle')
    
    def generate_phone(self):
        """生成 phone (手机) 图片"""
        fig, ax = self.create_base_figure(bg_color='#1a1a2e')
        
        # 手机外壳
        phone_body = FancyBboxPatch((3, 2), 4, 6.5,
                                   boxstyle="round,pad=0.2",
                                   facecolor='#1f2937',
                                   edgecolor='#6b7280',
                                   linewidth=4, zorder=2)
        ax.add_patch(phone_body)
        
        # 屏幕
        screen = Rectangle((3.3, 2.8), 3.4, 5,
                          facecolor='#0ea5e9',
                          edgecolor='#0284c7',
                          linewidth=2, zorder=3)
        ax.add_patch(screen)
        
        # 屏幕内容（模拟图标）
        icon_size = 0.5
        positions = [(4, 7), (5, 7), (6, 7),
                    (4, 6), (5, 6), (6, 6),
                    (4, 5), (5, 5), (6, 5)]
        for x, y in positions:
            icon = Rectangle((x-icon_size/2, y-icon_size/2), icon_size, icon_size,
                           facecolor='#e0f2fe',
                           edgecolor='#0284c7',
                           linewidth=1, zorder=4)
            ax.add_patch(icon)
        
        # 前置摄像头
        camera = Circle((5, 8.2), 0.15, color='#374151', zorder=4)
        ax.add_patch(camera)
        
        # Home按钮
        home_button = Circle((5, 2.4), 0.25, 
                            facecolor='#374151',
                            edgecolor='#6b7280',
                            linewidth=2, zorder=4)
        ax.add_patch(home_button)
        
        # 文字标签
        ax.text(5, 1.2, 'PHONE', fontsize=32, fontweight='bold',
               ha='center', va='center', color='#0ea5e9')
        ax.text(5, 0.5, '手机', fontsize=24,
               ha='center', va='center', color='#38bdf8')
        
        filepath = os.path.join(self.objects_dir, 'phone.png')
        self.save_and_close(fig, filepath, 'Phone')
    
    def generate_spanner(self):
        """生成 spanner (扳手) 图片"""
        fig, ax = self.create_base_figure(bg_color='#1a1a2e')
        
        # 扳手手柄（主体）
        handle = Rectangle((2, 4.2), 5, 1.2,
                          facecolor='#71717a',
                          edgecolor='#3f3f46',
                          linewidth=3, zorder=2)
        ax.add_patch(handle)
        
        # 扳手头部（开口部分）
        # 左侧上颚
        upper_jaw = Polygon([(7, 5.4), (8.5, 6.5), (8.5, 7), (7, 5.9)],
                           facecolor='#71717a',
                           edgecolor='#3f3f46',
                           linewidth=3, zorder=2)
        ax.add_patch(upper_jaw)
        
        # 左侧下颚
        lower_jaw = Polygon([(7, 4.6), (8.5, 3.5), (8.5, 3), (7, 4.1)],
                           facecolor='#71717a',
                           edgecolor='#3f3f46',
                           linewidth=3, zorder=2)
        ax.add_patch(lower_jaw)
        
        # 高光效果
        highlight1 = Rectangle((2.2, 4.9), 4.5, 0.3,
                              facecolor='#a1a1aa',
                              alpha=0.6, zorder=3)
        ax.add_patch(highlight1)
        
        # 手柄纹理
        for i in range(3, 7):
            ax.plot([i, i], [4.3, 5.3], color='#52525b', linewidth=2, alpha=0.5, zorder=3)
        
        # 文字标签
        ax.text(5, 1.5, 'SPANNER', fontsize=32, fontweight='bold',
               ha='center', va='center', color='#71717a')
        ax.text(5, 0.7, '扳手', fontsize=24,
               ha='center', va='center', color='#a1a1aa')
        
        filepath = os.path.join(self.objects_dir, 'spanner.png')
        self.save_and_close(fig, filepath, 'Spanner')
    
    # ========== 动作图片生成 ==========
    
    def generate_none(self):
        """生成 none (无动作) 图片"""
        fig, ax = self.create_base_figure(bg_color='#0a0e27')
        
        # 绘制禁止符号
        circle = Circle((5, 5), 2.5, 
                       facecolor='none',
                       edgecolor='#ef4444',
                       linewidth=6, zorder=2)
        ax.add_patch(circle)
        
        # 斜杠
        ax.plot([3.2, 6.8], [6.8, 3.2], color='#ef4444', linewidth=6, zorder=3)
        
        # 文字
        ax.text(5, 1.5, 'NONE', fontsize=38, fontweight='bold',
               ha='center', va='center', color='#ef4444')
        ax.text(5, 0.7, '无动作', fontsize=28,
               ha='center', va='center', color='#f87171')
        
        filepath = os.path.join(self.actions_dir, 'none.png')
        self.save_and_close(fig, filepath, 'None')
    
    def generate_hold(self):
        """生成 hold (握持) 图片"""
        fig, ax = self.create_base_figure(bg_color='#1a1a2e')
        
        # 绘制手掌
        palm = FancyBboxPatch((3.5, 3), 3, 4,
                             boxstyle="round,pad=0.15",
                             facecolor='#fbbf24',
                             edgecolor='#f59e0b',
                             linewidth=3, zorder=2)
        ax.add_patch(palm)
        
        # 绘制手指
        fingers_data = [
            (4, 7, 0.4, 1.5),   # 食指
            (4.8, 7.3, 0.4, 1.8),  # 中指
            (5.6, 7.2, 0.4, 1.6),  # 无名指
            (6.3, 6.8, 0.35, 1.2), # 小指
        ]
        
        for x, y, w, h in fingers_data:
            finger = FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#fbbf24',
                                   edgecolor='#f59e0b',
                                   linewidth=2, zorder=2)
            ax.add_patch(finger)
        
        # 拇指
        thumb = FancyBboxPatch((3, 5), 0.8, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#fbbf24',
                              edgecolor='#f59e0b',
                              linewidth=2, zorder=2)
        ax.add_patch(thumb)
        
        # 被握持的物体
        object_held = Circle((5, 4.5), 0.8,
                           facecolor='#3b82f6',
                           edgecolor='#1e40af',
                           linewidth=2, zorder=3)
        ax.add_patch(object_held)
        
        # 文字
        ax.text(5, 1.5, 'HOLD', fontsize=36, fontweight='bold',
               ha='center', va='center', color='#fbbf24')
        ax.text(5, 0.7, '握持', fontsize=28,
               ha='center', va='center', color='#fcd34d')
        
        filepath = os.path.join(self.actions_dir, 'hold.png')
        self.save_and_close(fig, filepath, 'Hold')
    
    def generate_tap(self):
        """生成 tap (轻敲) 图片"""
        fig, ax = self.create_base_figure(bg_color='#1a1a2e')
        
        # 绘制手指
        finger = FancyBboxPatch((4, 5), 2, 3,
                               boxstyle="round,pad=0.1",
                               facecolor='#fbbf24',
                               edgecolor='#f59e0b',
                               linewidth=3, zorder=2)
        ax.add_patch(finger)
        
        # 指尖
        fingertip = Circle((5, 5), 0.6,
                          facecolor='#fcd34d',
                          edgecolor='#f59e0b',
                          linewidth=2, zorder=3)
        ax.add_patch(fingertip)
        
        # 表面
        surface = Rectangle((2, 3.5), 6, 0.5,
                           facecolor='#4b5563',
                           edgecolor='#1f2937',
                           linewidth=3, zorder=1)
        ax.add_patch(surface)
        
        # 冲击波效果（表示轻敲）
        for i, radius in enumerate([0.8, 1.2, 1.6]):
            wave = Circle((5, 4.2), radius,
                         facecolor='none',
                         edgecolor='#10b981',
                         linewidth=3,
                         alpha=0.7 - i*0.2,
                         zorder=4)
            ax.add_patch(wave)
        
        # 向下箭头
        arrow = patches.FancyArrow(5, 8.5, 0, -0.8,
                                  width=0.5,
                                  head_width=0.8,
                                  head_length=0.4,
                                  facecolor='#10b981',
                                  edgecolor='#059669',
                                  linewidth=2, zorder=5)
        ax.add_patch(arrow)
        
        # 文字
        ax.text(5, 1.5, 'TAP', fontsize=36, fontweight='bold',
               ha='center', va='center', color='#10b981')
        ax.text(5, 0.7, '轻敲', fontsize=28,
               ha='center', va='center', color='#34d399')
        
        filepath = os.path.join(self.actions_dir, 'tap.png')
        self.save_and_close(fig, filepath, 'Tap')
    
    def generate_hammer(self):
        """生成 hammer (锤击) 图片"""
        fig, ax = self.create_base_figure(bg_color='#1a1a2e')
        
        # 绘制拳头（锤击姿势）
        # 手掌
        fist = FancyBboxPatch((3.5, 4.5), 3, 2.5,
                             boxstyle="round,pad=0.1",
                             facecolor='#fbbf24',
                             edgecolor='#f59e0b',
                             linewidth=3, zorder=2)
        ax.add_patch(fist)
        
        # 手臂
        arm = Rectangle((4, 7), 2, 1.5,
                       facecolor='#fcd34d',
                       edgecolor='#f59e0b',
                       linewidth=2, zorder=1)
        ax.add_patch(arm)
        
        # 表面
        surface = Rectangle((2, 3), 6, 0.6,
                           facecolor='#4b5563',
                           edgecolor='#1f2937',
                           linewidth=3, zorder=1)
        ax.add_patch(surface)
        
        # 强烈冲击效果
        # 多重冲击波
        for i, radius in enumerate([0.9, 1.4, 1.9, 2.4]):
            wave = Circle((5, 3.8), radius,
                         facecolor='none',
                         edgecolor='#ef4444',
                         linewidth=4,
                         alpha=0.8 - i*0.15,
                         zorder=4)
            ax.add_patch(wave)
        
        # 爆炸星形效果
        star_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for angle in star_angles:
            x_end = 5 + 1.5 * np.cos(angle)
            y_end = 3.8 + 1.5 * np.sin(angle)
            ax.plot([5, x_end], [3.8, y_end], 
                   color='#fef08a', linewidth=3, alpha=0.8, zorder=5)
        
        # 向下双箭头（表示用力）
        for offset in [-0.8, 0.8]:
            arrow = patches.FancyArrow(5+offset, 8.8, 0, -0.6,
                                      width=0.4,
                                      head_width=0.6,
                                      head_length=0.3,
                                      facecolor='#ef4444',
                                      edgecolor='#dc2626',
                                      linewidth=2, zorder=5)
            ax.add_patch(arrow)
        
        # 文字
        ax.text(5, 1.5, 'HAMMER', fontsize=34, fontweight='bold',
               ha='center', va='center', color='#ef4444')
        ax.text(5, 0.7, '锤击', fontsize=28,
               ha='center', va='center', color='#f87171')
        
        filepath = os.path.join(self.actions_dir, 'hammer.png')
        self.save_and_close(fig, filepath, 'Hammer')
    
    def generate_finger_press(self):
        """生成 finger_press (指压) 图片"""
        fig, ax = self.create_base_figure(bg_color='#1a1a2e')
        
        # 绘制手指（按压姿势）
        finger = FancyBboxPatch((3.8, 5.5), 2.4, 3,
                               boxstyle="round,pad=0.12",
                               facecolor='#fbbf24',
                               edgecolor='#f59e0b',
                               linewidth=3, zorder=2)
        ax.add_patch(finger)
        
        # 指尖压痕
        fingertip = patches.Ellipse((5, 5.3), 1.2, 0.6,
                                   facecolor='#f59e0b',
                                   edgecolor='#d97706',
                                   linewidth=2, zorder=3)
        ax.add_patch(fingertip)
        
        # 表面
        surface = Rectangle((2, 4), 6, 0.8,
                           facecolor='#4b5563',
                           edgecolor='#1f2937',
                           linewidth=3, zorder=1)
        ax.add_patch(surface)
        
        # 压力指示（变形效果）
        # 表面凹陷
        depression = patches.Ellipse((5, 4.4), 1.8, 0.3,
                                    facecolor='#374151',
                                    alpha=0.6, zorder=2)
        ax.add_patch(depression)
        
        # 压力线（表示持续按压）
        for i in range(3):
            y_pos = 5.5 + i * 0.4
            ax.plot([3.3, 3.3], [y_pos, y_pos + 0.2],
                   color='#ef4444', linewidth=3, alpha=0.7, zorder=4)
            ax.plot([6.7, 6.7], [y_pos, y_pos + 0.2],
                   color='#ef4444', linewidth=3, alpha=0.7, zorder=4)
        
        # 压力波纹（持续压力）
        for i, radius in enumerate([1.0, 1.5, 2.0]):
            wave = patches.Ellipse((5, 4.4), radius*1.5, radius*0.5,
                                  facecolor='none',
                                  edgecolor='#8b5cf6',
                                  linewidth=2,
                                  alpha=0.6 - i*0.15,
                                  zorder=4)
            ax.add_patch(wave)
        
        # 向下箭头（表示按压）
        arrow = patches.FancyArrow(5, 9, 0, -0.8,
                                  width=0.6,
                                  head_width=1.0,
                                  head_length=0.4,
                                  facecolor='#8b5cf6',
                                  edgecolor='#7c3aed',
                                  linewidth=2, zorder=5)
        ax.add_patch(arrow)
        
        # 文字
        ax.text(5, 1.8, 'FINGER PRESS', fontsize=28, fontweight='bold',
               ha='center', va='center', color='#8b5cf6')
        ax.text(5, 0.9, '指压', fontsize=28,
               ha='center', va='center', color='#a78bfa')
        
        filepath = os.path.join(self.actions_dir, 'finger_press.png')
        self.save_and_close(fig, filepath, 'Finger Press')
    
    def generate_all(self):
        """生成所有参考图片"""
        print("\n" + "="*70)
        print("🎨 开始生成参考图片")
        print("="*70)
        
        print("\n📦 生成物品图片:")
        self.generate_empty()
        self.generate_ball()
        self.generate_bottle()
        self.generate_phone()
        self.generate_spanner()
        
        print("\n🎬 生成动作图片:")
        self.generate_none()
        self.generate_hold()
        self.generate_tap()
        self.generate_hammer()
        self.generate_finger_press()
        
        print("\n" + "="*70)
        print("✅ 所有参考图片生成完成！")
        print("="*70)
        print(f"\n📁 输出目录: {self.output_dir}")
        print(f"   - 物品图片: {self.objects_dir}")
        print(f"   - 动作图片: {self.actions_dir}")
        print("\n💡 提示: 这些图片将在实时推理系统中作为参考显示")
        print("="*70 + "\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成参考图片')
    parser.add_argument('--output-dir', type=str, default='reference_images',
                       help='输出目录 (默认: reference_images)')
    parser.add_argument('--size', type=int, nargs=2, default=[400, 400],
                       help='图片尺寸 (宽 高) (默认: 400 400)')
    parser.add_argument('--dpi', type=int, default=100,
                       help='图片DPI (默认: 100)')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = ReferenceImageGenerator(
        output_dir=args.output_dir,
        img_size=tuple(args.size),
        dpi=args.dpi
    )
    
    # 生成所有图片
    generator.generate_all()


if __name__ == '__main__':
    main()

