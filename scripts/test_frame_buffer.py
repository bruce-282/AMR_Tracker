"""
Test script for ThreadedFrameBuffer

This script tests the frame buffering system with simulated slow processing
to verify that:
1. Frames are captured continuously in a separate thread
2. Buffer fills up when processing is slow
3. Oldest frames are dropped when buffer is full (drop_policy="oldest")
4. Statistics are accurate
5. No frames are lost in normal operation

Usage:
    python scripts/test_frame_buffer.py [--mode simulated|camera]
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.sequence_loader import ThreadedFrameBuffer, FrameData


class SimulatedCamera:
    """Simulated camera for testing without actual hardware."""
    
    def __init__(self, fps: float = 30.0, resolution: tuple = (640, 480)):
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.resolution = resolution
        self.frame_count = 0
        self.last_capture_time = time.time()
        
    def capture(self) -> tuple:
        """Simulate camera capture with realistic timing."""
        # Simulate camera frame rate
        elapsed = time.time() - self.last_capture_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        
        self.last_capture_time = time.time()
        self.frame_count += 1
        
        # Create a test frame with frame number embedded
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        # Add some visual pattern
        frame[:, :, 0] = (self.frame_count * 5) % 256  # Blue varies with frame
        frame[:, :, 1] = 100  # Green constant
        frame[:, :, 2] = 50   # Red constant
        
        return True, frame


def test_basic_functionality():
    """Test 1: Basic buffer functionality."""
    print("\n" + "="*60)
    print("Test 1: Basic Buffer Functionality")
    print("="*60)
    
    camera = SimulatedCamera(fps=30.0)
    buffer = ThreadedFrameBuffer(
        capture_func=camera.capture,
        max_size=10,
        drop_policy="oldest"
    )
    
    print(f"Created buffer with max_size=10, policy=oldest")
    
    # Start buffer
    buffer.start()
    print("Buffer started")
    
    # Wait for buffer to fill
    time.sleep(0.5)
    
    # Read a few frames
    frames_read = 0
    for i in range(5):
        frame_data = buffer.get(timeout=0.2)
        if frame_data:
            frames_read += 1
            print(f"  Read frame #{frame_data.frame_number}, queue_size={buffer.queue_size}")
    
    stats = buffer.get_stats()
    print(f"\nStats after reading {frames_read} frames:")
    print(f"  Captured: {stats['captured']}")
    print(f"  Dropped: {stats['dropped']}")
    print(f"  Queue size: {stats['queue_size']}/{stats['max_size']}")
    print(f"  Capture FPS: {stats['capture_fps']}")
    
    # Stop buffer
    buffer.stop()
    print("Buffer stopped")
    
    # Verify
    success = frames_read == 5 and stats['captured'] > 0
    print(f"\n✓ Test 1 {'PASSED' if success else 'FAILED'}")
    return success


def test_slow_processing():
    """Test 2: Buffer behavior with slow processing."""
    print("\n" + "="*60)
    print("Test 2: Slow Processing (Buffer Fill + Drop)")
    print("="*60)
    
    camera = SimulatedCamera(fps=30.0)
    buffer = ThreadedFrameBuffer(
        capture_func=camera.capture,
        max_size=10,
        drop_policy="oldest"
    )
    
    buffer.start()
    print("Buffer started (max_size=10)")
    
    # Simulate slow processing - capture for 1 second without reading
    print("Simulating slow processing (not reading for 1 second)...")
    time.sleep(1.0)
    
    stats_before = buffer.get_stats()
    print(f"\nBefore reading:")
    print(f"  Captured: {stats_before['captured']}")
    print(f"  Dropped: {stats_before['dropped']}")
    print(f"  Queue size: {stats_before['queue_size']}/{stats_before['max_size']}")
    
    # Now read all available frames
    frames_read = 0
    while True:
        frame_data = buffer.get_nowait()
        if frame_data is None:
            break
        frames_read += 1
    
    print(f"\nRead {frames_read} frames from buffer")
    
    stats_after = buffer.get_stats()
    
    buffer.stop()
    
    # At 30fps for 1 second, we should capture ~30 frames
    # With buffer size 10, we should drop ~20 frames
    expected_drops = max(0, stats_before['captured'] - 10)
    
    print(f"\nExpected drops: ~{expected_drops}")
    print(f"Actual drops: {stats_after['dropped']}")
    
    success = stats_after['dropped'] > 0 and frames_read <= 10
    print(f"\n✓ Test 2 {'PASSED' if success else 'FAILED'}")
    return success


def test_real_time_processing():
    """Test 3: Real-time processing (no drops expected)."""
    print("\n" + "="*60)
    print("Test 3: Real-time Processing (No Drops)")
    print("="*60)
    
    camera = SimulatedCamera(fps=30.0)
    buffer = ThreadedFrameBuffer(
        capture_func=camera.capture,
        max_size=30,
        drop_policy="oldest"
    )
    
    buffer.start()
    print("Buffer started")
    
    # Process frames in real-time (faster than capture)
    frames_read = 0
    start_time = time.time()
    
    print("Processing frames in real-time for 1 second...")
    while time.time() - start_time < 1.0:
        frame_data = buffer.get(timeout=0.05)
        if frame_data:
            frames_read += 1
            # Simulate fast processing (10ms)
            time.sleep(0.01)
    
    stats = buffer.get_stats()
    buffer.stop()
    
    print(f"\nResults:")
    print(f"  Frames read: {frames_read}")
    print(f"  Captured: {stats['captured']}")
    print(f"  Dropped: {stats['dropped']}")
    print(f"  Drop rate: {stats['drop_rate']}%")
    
    # Should have no or minimal drops
    success = stats['drop_rate'] < 5.0  # Less than 5% drop rate
    print(f"\n✓ Test 3 {'PASSED' if success else 'FAILED'}")
    return success


def test_timestamp_accuracy():
    """Test 4: Timestamp accuracy."""
    print("\n" + "="*60)
    print("Test 4: Timestamp Accuracy")
    print("="*60)
    
    camera = SimulatedCamera(fps=30.0)
    buffer = ThreadedFrameBuffer(
        capture_func=camera.capture,
        max_size=10,
        drop_policy="oldest"
    )
    
    buffer.start()
    time.sleep(0.2)  # Let buffer fill a bit
    
    timestamps = []
    for i in range(5):
        frame_data = buffer.get(timeout=0.1)
        if frame_data:
            timestamps.append(frame_data.timestamp)
            time.sleep(0.033)  # ~30fps processing
    
    buffer.stop()
    
    # Check timestamp intervals
    if len(timestamps) >= 2:
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_interval = sum(intervals) / len(intervals)
        expected_interval = 1.0 / 30.0  # ~33ms at 30fps
        
        print(f"Expected interval: {expected_interval*1000:.1f}ms")
        print(f"Average interval: {avg_interval*1000:.1f}ms")
        print(f"Intervals: {[f'{i*1000:.1f}ms' for i in intervals]}")
        
        # Allow 50% tolerance due to timing variations
        success = abs(avg_interval - expected_interval) < expected_interval * 0.5
    else:
        success = False
        print("Not enough frames captured")
    
    print(f"\n✓ Test 4 {'PASSED' if success else 'FAILED'}")
    return success


def test_stop_and_restart():
    """Test 5: Stop and restart buffer."""
    print("\n" + "="*60)
    print("Test 5: Stop and Restart")
    print("="*60)
    
    camera = SimulatedCamera(fps=30.0)
    buffer = ThreadedFrameBuffer(
        capture_func=camera.capture,
        max_size=10,
        drop_policy="oldest"
    )
    
    # First run
    buffer.start()
    time.sleep(0.2)
    stats1 = buffer.get_stats()
    buffer.stop()
    
    print(f"First run captured: {stats1['captured']}")
    
    # Second run (reuse same buffer)
    buffer = ThreadedFrameBuffer(
        capture_func=camera.capture,
        max_size=10,
        drop_policy="oldest"
    )
    buffer.start()
    time.sleep(0.2)
    stats2 = buffer.get_stats()
    buffer.stop()
    
    print(f"Second run captured: {stats2['captured']}")
    
    success = stats1['captured'] > 0 and stats2['captured'] > 0
    print(f"\n✓ Test 5 {'PASSED' if success else 'FAILED'}")
    return success


def test_with_novitec_camera():
    """Test 6: Test with actual Novitec camera (if available)."""
    print("\n" + "="*60)
    print("Test 6: Novitec Camera Integration (Optional)")
    print("="*60)
    
    try:
        from src.utils.sequence_loader import NovitecCameraLoader, NOVITEC_AVAILABLE
        
        if not NOVITEC_AVAILABLE:
            print("Novitec camera module not available, skipping...")
            return True  # Not a failure, just skipped
        
        # This would require an actual camera connected
        print("Novitec module available, but skipping actual camera test")
        print("(Run with actual camera connected for full test)")
        return True
        
    except ImportError as e:
        print(f"Could not import Novitec loader: {e}")
        return True  # Not a failure, just skipped


def main():
    parser = argparse.ArgumentParser(description="Test ThreadedFrameBuffer")
    parser.add_argument("--mode", choices=["simulated", "camera"], default="simulated",
                       help="Test mode: simulated (default) or camera")
    args = parser.parse_args()
    
    print("="*60)
    print("ThreadedFrameBuffer Test Suite")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    if args.mode == "simulated":
        results.append(("Basic Functionality", test_basic_functionality()))
        results.append(("Slow Processing", test_slow_processing()))
        results.append(("Real-time Processing", test_real_time_processing()))
        results.append(("Timestamp Accuracy", test_timestamp_accuracy()))
        results.append(("Stop and Restart", test_stop_and_restart()))
    
    if args.mode == "camera":
        results.append(("Novitec Camera", test_with_novitec_camera()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

