"""
Serial Data Receiver for 16x16 Pressure Sensor Matrix
Receives 256 bytes per frame via UART at 115200 baud
强力降噪版 - 专门针对高帧率噪声问题
通过增加多帧平均数量和优化滤波参数来大幅降低噪声
"""

import serial
import numpy as np
import time
from collections import deque
import threading
import queue
from scipy import ndimage
from scipy.ndimage import zoom


class PressureSensorReceiver:
    """
    Handles serial communication with the FPGA pressure sensor system
    强力降噪版本 - 针对高帧率传输优化
    """
    
    def __init__(self, port='COM3', baudrate=115200, timeout=1.0,
                 enable_noise_reduction=True, 
                 noise_threshold=15,           # 适中的阈值
                 temporal_smoothing=0.7,       # 增强时域平滑
                 spatial_smoothing=0.8,        # 增强空间平滑
                 multi_frame_average=True, 
                 average_frames=10,            # 增加到10帧！关键参数
                 display_fps_limit=20):        # 新增：限制显示帧率
        """
        Initialize serial receiver with aggressive noise reduction
        
        Args:
            port: Serial port name
            baudrate: Baud rate (default: 115200)
            timeout: Read timeout in seconds
            enable_noise_reduction: Enable noise reduction (default: True)
            noise_threshold: Noise threshold (default: 15)
            temporal_smoothing: Temporal smoothing (default: 0.7, 增强了！)
            spatial_smoothing: Spatial smoothing (default: 0.8, 增强了！)
            multi_frame_average: Enable multi-frame averaging (default: True)
            average_frames: Frames to average (default: 10, 增加了！)
            display_fps_limit: Limit display FPS (default: 20)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.is_running = False
        self.data_queue = queue.Queue(maxsize=100)
        
        # ========== 多帧平均参数（增强！）==========
        self.multi_frame_average = multi_frame_average
        self.average_frames = average_frames  # 增加到10帧
        self.frame_buffer = deque(maxlen=average_frames)
        
        # ========== 帧率控制（新增！）==========
        self.display_fps_limit = display_fps_limit
        self.last_display_time = 0
        self.min_display_interval = 1.0 / display_fps_limit if display_fps_limit > 0 else 0
        
        # Frame parameters
        self.frame_size = 256  # 16x16 matrix
        self.matrix_shape = (16, 16)
        
        # Statistics
        self.frames_received = 0
        self.frames_displayed = 0  # 新增：实际显示的帧数
        self.errors_count = 0
        self.last_frame_time = 0
        self.fps = 0
        self.display_fps = 0  # 新增：显示帧率
        
        # ========== 降噪参数（增强！）==========
        self.enable_noise_reduction = enable_noise_reduction
        self.noise_threshold = noise_threshold
        self.temporal_smoothing = temporal_smoothing  # 增强
        self.spatial_smoothing = spatial_smoothing    # 增强
        
        # 背景消除
        self.background_frame = None
        self.background_calibrated = False
        self.background_samples = []
        self.background_sample_count = 30  # 增加到30帧以获得更准确的背景
        
        # 时间滤波
        self.prev_frame = None
        
        # 形态学降噪
        self.morph_kernel = np.ones((3, 3), dtype=np.uint8)  # 增大核
        
        print(f"🎯 强力降噪模式已启用")
        print(f"   多帧平均: {self.average_frames} 帧")
        print(f"   显示帧率限制: {self.display_fps_limit} FPS")
        print(f"   时域平滑: {self.temporal_smoothing}")
        print(f"   空域平滑: {self.spatial_smoothing}")
    
    def upscale_frame(self, frame, target_size=256):
        """高质量插值"""
        if frame.shape[0] == target_size and frame.shape[1] == target_size:
            return frame
        
        zoom_factor = target_size / frame.shape[0]
        upscaled = zoom(frame, zoom_factor, order=3, mode='nearest')
        upscaled = ndimage.gaussian_filter(upscaled, sigma=0.5)
        
        return upscaled.astype(np.uint8)
        
    def connect(self):
        """Connect to serial port"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            print(f"✅ Connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"❌ Error connecting to serial port: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial port"""
        self.is_running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial connection closed")
    
    def calibrate_background(self, num_samples=30):
        """
        校准背景 - 增加采样数量以获得更准确的背景
        """
        print(f"\n🔧 开始背景校准...")
        print(f"⚠️  请确保传感器上没有任何物体!")
        print(f"   将采集 {num_samples} 帧作为背景基准（比之前更多以提高准确度）...")
        
        self.background_samples = []
        
        for i in range(num_samples):
            frame = self._read_raw_frame()
            if frame is not None:
                self.background_samples.append(frame.astype(np.float32))
                print(f"   采集进度: {i+1}/{num_samples}", end='\r')
                time.sleep(0.05)
        
        if len(self.background_samples) > 0:
            # 使用中位数作为背景（更鲁棒）
            self.background_frame = np.median(self.background_samples, axis=0).astype(np.uint8)
            self.background_calibrated = True
            print(f"\n✅ 背景校准完成!")
            print(f"   背景平均值: {self.background_frame.mean():.1f}")
            print(f"   背景最大值: {self.background_frame.max()}")
            print(f"   背景标准差: {self.background_frame.std():.1f}")
        else:
            print(f"\n❌ 背景校准失败!")
            self.background_calibrated = False
    
    def reset_background(self):
        """重置背景校准"""
        self.background_frame = None
        self.background_calibrated = False
        self.background_samples = []
        print("🔄 背景已重置")
    
    def _read_raw_frame(self):
        """读取原始帧"""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None
        
        try:
            data = self.serial_conn.read(self.frame_size)
            
            if len(data) != self.frame_size:
                self.errors_count += 1
                return None
            
            frame = np.frombuffer(data, dtype=np.uint8)
            frame = frame.reshape(self.matrix_shape)
            
            return frame
            
        except Exception as e:
            self.errors_count += 1
            return None
    
    def _apply_noise_reduction(self, frame):
        """
        应用降噪算法 - 增强版
        多级滤波：背景消除 → 阈值过滤 → 双边滤波 → 时域滤波 → 形态学处理 → 对比度增强
        """
        if not self.enable_noise_reduction:
            return frame
        
        frame_float = frame.astype(np.float32)
        
        # 1. 背景消除
        if self.background_calibrated and self.background_frame is not None:
            frame_float = frame_float - self.background_frame.astype(np.float32)
            frame_float = np.maximum(frame_float, 0)
        
        # 2. 噪声阈值过滤
        frame_float = np.where(frame_float < self.noise_threshold, 0, frame_float)
        
        # 3. 增强的空间滤波
        if self.spatial_smoothing > 0:
            # 先使用中值滤波去除椒盐噪声
            from scipy.ndimage import median_filter
            frame_float = median_filter(frame_float, size=3)
            
            # 再使用高斯滤波平滑
            sigma = self.spatial_smoothing
            frame_float = ndimage.gaussian_filter(frame_float, sigma=sigma)
        
        # 4. 增强的时间滤波（EMA）
        if self.prev_frame is not None and self.temporal_smoothing > 0:
            alpha = 1.0 - self.temporal_smoothing
            frame_float = alpha * frame_float + self.temporal_smoothing * self.prev_frame
        
        self.prev_frame = frame_float.copy()
        
        # 5. 轮廓增强
        if np.any(frame_float > self.noise_threshold):
            from scipy.ndimage import sobel
            sx = sobel(frame_float, axis=0, mode='constant')
            sy = sobel(frame_float, axis=1, mode='constant')
            edge_magnitude = np.sqrt(sx**2 + sy**2)
            
            if edge_magnitude.max() > 0:
                edge_magnitude = edge_magnitude / edge_magnitude.max() * 40
            
            mask = frame_float > self.noise_threshold
            frame_float[mask] = frame_float[mask] + edge_magnitude[mask] * 0.25
        
        # 6. 形态学操作
        frame_uint8 = np.clip(frame_float, 0, 255).astype(np.uint8)
        
        if np.count_nonzero(frame_uint8) > 3:
            binary = (frame_uint8 > self.noise_threshold).astype(np.uint8)
            
            from scipy.ndimage import binary_opening, binary_closing
            # 开运算去除小噪点
            binary_clean = binary_opening(binary, structure=self.morph_kernel, iterations=1)
            # 闭运算填补孔洞
            binary_clean = binary_closing(binary_clean, structure=self.morph_kernel, iterations=1)
            
            frame_clean = frame_uint8 * binary_clean
        else:
            frame_clean = frame_uint8
        
        # 7. 对比度增强
        if np.any(frame_clean > 0):
            non_zero_mask = frame_clean > 0
            values = frame_clean[non_zero_mask]
            
            if len(values) > 0:
                v_min, v_max = values.min(), values.max()
                if v_max > v_min and v_max > self.noise_threshold * 1.5:
                    stretched = (values - v_min) / (v_max - v_min) * (220) + 35
                    frame_clean[non_zero_mask] = np.clip(stretched, 0, 255).astype(np.uint8)
        
        return frame_clean
    
    def read_frame(self):
        """Read one complete frame with noise reduction"""
        frame = self._read_raw_frame()
        
        if frame is None:
            return None
        
        # 更新统计
        self.frames_received += 1
        current_time = time.time()
        if self.last_frame_time > 0:
            time_diff = current_time - self.last_frame_time
            self.fps = 1.0 / time_diff if time_diff > 0 else 0
        self.last_frame_time = current_time
        
        # 应用降噪
        frame_clean = self._apply_noise_reduction(frame)
        
        # 应用多帧平均（关键！）
        if self.multi_frame_average:
            frame_clean = self._apply_multi_frame_average(frame_clean)
        
        return frame_clean
    
    def _apply_multi_frame_average(self, frame):
        """
        应用多帧平均 - 增强版
        使用更多帧进行平均，显著降低噪声
        """
        self.frame_buffer.append(frame.astype(np.float32))
        
        # 如果缓冲区未满，返回当前帧
        if len(self.frame_buffer) < self.average_frames:
            return frame
        
        # 计算平均值（关键降噪步骤）
        averaged_frame = np.mean(self.frame_buffer, axis=0)
        
        # 可选：加权平均，最新帧权重稍高
        # weights = np.linspace(0.8, 1.2, len(self.frame_buffer))
        # averaged_frame = np.average(self.frame_buffer, axis=0, weights=weights)
        
        averaged_frame = np.clip(averaged_frame, 0, 255).astype(np.uint8)
        
        return averaged_frame
    
    def start_continuous_reading(self):
        """Start continuous reading"""
        if self.is_running:
            print("Already running")
            return
        
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return
        
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        print("Started continuous reading")
    
    def stop_continuous_reading(self):
        """Stop continuous reading"""
        self.is_running = False
        if hasattr(self, 'read_thread'):
            self.read_thread.join(timeout=2.0)
        print("Stopped continuous reading")
    
    def _read_loop(self):
        """Internal read loop with frame rate control"""
        while self.is_running:
            frame = self.read_frame()
            if frame is not None:
                # 帧率控制：只在足够时间间隔后才放入队列
                current_time = time.time()
                if current_time - self.last_display_time >= self.min_display_interval:
                    try:
                        # 清空队列，只保留最新帧
                        while not self.data_queue.empty():
                            try:
                                self.data_queue.get_nowait()
                            except:
                                break
                        
                        self.data_queue.put_nowait({
                            'frame': frame,
                            'timestamp': current_time,
                            'frame_number': self.frames_received
                        })
                        
                        self.frames_displayed += 1
                        self.last_display_time = current_time
                        
                        # 计算显示帧率
                        if self.frames_displayed > 1:
                            self.display_fps = self.frames_displayed / (current_time - self.start_time) if hasattr(self, 'start_time') else 0
                        
                    except queue.Full:
                        pass
                # else: 跳过这一帧，不显示
    
    def get_latest_frame(self, timeout=1.0):
        """Get the latest frame from the queue"""
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
        
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_statistics(self):
        """Get receiver statistics"""
        return {
            'frames_received': self.frames_received,
            'frames_displayed': self.frames_displayed,
            'errors_count': self.errors_count,
            'fps': self.fps,
            'display_fps': self.display_fps,
            'queue_size': self.data_queue.qsize(),
            'average_frames': self.average_frames
        }


def test_receiver():
    """Test function"""
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("🎯 压力传感器接收器测试 (强力降噪版)")
    print("=" * 70)
    print()
    print("此版本专门针对高帧率噪声优化")
    print("特点：")
    print("  - 10帧滑动平均（降噪率约68%）")
    print("  - 增强的时域和空域滤波")
    print("  - 显示帧率限制在20 FPS")
    print("  - 更准确的背景校准（30帧）")
    print()
    
    # Create receiver with strong noise reduction
    receiver = PressureSensorReceiver(
        port='COM3',
        enable_noise_reduction=True,
        noise_threshold=15,
        temporal_smoothing=0.7,
        spatial_smoothing=0.8,
        multi_frame_average=True,
        average_frames=10,     # 10帧平均
        display_fps_limit=20   # 限制显示帧率
    )
    
    if not receiver.connect():
        print("Failed to connect")
        return
    
    # 背景校准
    print("\n准备进行背景校准...")
    print("⚠️  请确保传感器上没有任何物体!")
    input("按回车键开始校准...")
    receiver.calibrate_background(num_samples=30)
    
    # Setup visualization
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('强力降噪版 - 10帧平均 + 帧率限制', fontsize=16, fontweight='bold')
    
    # 热力图
    im1 = ax1.imshow(np.zeros((16, 16)), cmap='hot', vmin=0, vmax=255)
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('压力分布（强力降噪）')
    
    # 直方图
    ax2.set_title('压力分布直方图')
    
    # 统计信息
    ax3.axis('off')
    stats_text = ax3.text(0.1, 0.5, '', fontsize=10, family='monospace',
                         verticalalignment='center')
    
    # 压力曲线
    ax4.set_title('压力变化曲线')
    ax4.set_xlabel('时间')
    ax4.set_ylabel('压力值')
    ax4.grid(True, alpha=0.3)
    
    pressure_history = []
    max_history = []
    
    receiver.start_continuous_reading()
    
    print("\n✅ 测试开始!")
    print("   现在请按压传感器，观察降噪效果")
    print("   按 Ctrl+C 退出")
    print("=" * 70 + "\n")
    
    try:
        while True:
            frame_data = receiver.get_latest_frame(timeout=1.0)
            if frame_data:
                frame = frame_data['frame']
                
                # Update heatmap
                im1.set_data(frame)
                ax1.set_title(f"压力分布 - 帧#{frame_data['frame_number']}")
                
                # Update histogram
                ax2.clear()
                ax2.hist(frame.flatten(), bins=30, color='orange', alpha=0.7)
                ax2.set_title('压力分布直方图')
                
                # Update statistics
                stats = receiver.get_statistics()
                stats_info = f"""
📊 统计信息
{'='*40}
接收帧数: {stats['frames_received']}
显示帧数: {stats['frames_displayed']}
接收FPS:  {stats['fps']:.1f}
显示FPS:  {stats['display_fps']:.1f}
错误计数: {stats['errors_count']}

📈 当前帧信息
{'='*40}
最小值:   {frame.min()}
最大值:   {frame.max()}
平均值:   {frame.mean():.1f}
非零点:   {np.count_nonzero(frame)}

🎛️ 强力降噪设置
{'='*40}
多帧平均: ✅ {stats['average_frames']}帧
时域平滑: {receiver.temporal_smoothing}
空域平滑: {receiver.spatial_smoothing}
帧率限制: {receiver.display_fps_limit} FPS
背景校准: {'✅' if receiver.background_calibrated else '❌'}

理论降噪率: ~68% (√10 = 3.16倍)
                """
                stats_text.set_text(stats_info)
                
                # Update curve
                pressure_history.append(frame.mean())
                max_history.append(frame.max())
                if len(pressure_history) > 100:
                    pressure_history.pop(0)
                    max_history.pop(0)
                
                ax4.clear()
                ax4.plot(pressure_history, label='平均压力', color='blue', linewidth=2)
                ax4.plot(max_history, label='最大压力', color='red', linewidth=2)
                ax4.set_title('压力变化曲线（应该很平滑）')
                ax4.set_xlabel('帧序号')
                ax4.set_ylabel('压力值')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.pause(0.01)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  停止测试...")
    
    finally:
        receiver.stop_continuous_reading()
        receiver.disconnect()
        plt.close()
        print("✅ 测试结束")


if __name__ == '__main__':
    test_receiver()

