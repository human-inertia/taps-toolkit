"""TAPS Collection Layer — Tap Strap 2 SDK Integration"""

import time
import threading
from datetime import datetime, timezone
from tapsdk import TapSDK, TapInputMode


class TapStrap2:
    """Wrapper around Tap Python SDK for raw sensor data collection."""

    FINGERS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

    def __init__(self, sensitivity=None):
        """
        Args:
            sensitivity: [finger_accel, imu_gyro, imu_accel]
                finger_accel: 0=±16G(default), 1=±2G, 2=±4G, 3=±8G, 4=±16G
                imu_gyro: 0=±500dps(default), 1=±125, 2=±250, 3=±500, 4=±1000, 5=±2000
                imu_accel: 0=±4G(default), 1=±2G, 2=±4G, 3=±8G, 4=±16G
        """
        self.sensitivity = sensitivity or [0, 0, 0]
        self.sdk = None
        self.connected_devices = {}
        self.raw_data_callback = None
        self.tap_event_callback = None
        self.connection_callback = None
        self._running = False

        # Scale factors for converting raw values to physical units
        self._finger_accel_scale = [31.25, 3.91, 7.81, 15.62, 31.25]  # mg/LSB
        self._imu_gyro_scale = [17.5, 4.375, 8.75, 17.5, 35.0, 70.0]  # mdps/LSB
        self._imu_accel_scale = [0.122, 0.061, 0.122, 0.244, 0.488]  # mg/LSB

    def connect(self):
        """Initialize SDK and register callbacks."""
        self.sdk = TapSDK()
        self.sdk.register_connection_events(self._on_connect)
        self.sdk.register_disconnection_events(self._on_disconnect)
        self.sdk.register_raw_data_events(self._on_raw_data)
        self._running = True

    def _on_connect(self, identifier, name, fw):
        self.connected_devices[identifier] = {
            "name": name,
            "firmware": fw,
            "connected_at": datetime.now(timezone.utc).isoformat()
        }
        print(f"[TAPS] Connected: {name} (FW: {fw})")
        # Switch to raw sensor mode
        self.sdk.set_input_mode(
            TapInputMode("raw", sensitivity=self.sensitivity),
            identifier
        )
        print(f"[TAPS] Raw sensor mode active. Sensitivity: {self.sensitivity}")
        if self.connection_callback:
            self.connection_callback(identifier, name, fw)

    def _on_disconnect(self, identifier):
        name = self.connected_devices.pop(identifier, {}).get("name", "unknown")
        print(f"[TAPS] Disconnected: {name}")

    def _on_raw_data(self, identifier, raw_sensor_data):
        """Called by SDK for each raw sensor data packet."""
        if self.raw_data_callback:
            self.raw_data_callback(identifier, raw_sensor_data)

    def on_raw_data(self, callback):
        """Register callback for raw sensor data.
        
        callback(identifier, raw_sensor_data) where raw_sensor_data has:
            .timestamp - device clock ms
            .type - 'Device' (finger accels) or 'IMU' (thumb imu)
            .GetPoint(index) - returns Point3(x,y,z) for each channel
        """
        self.raw_data_callback = callback

    def on_connection(self, callback):
        self.connection_callback = callback

    def get_device_info(self):
        return dict(self.connected_devices)

    def get_scale_factors(self):
        """Return current scale factors based on sensitivity settings."""
        return {
            "finger_accel_mg_per_lsb": self._finger_accel_scale[self.sensitivity[0]],
            "imu_gyro_mdps_per_lsb": self._imu_gyro_scale[self.sensitivity[1]],
            "imu_accel_mg_per_lsb": self._imu_accel_scale[self.sensitivity[2]],
        }

    def send_haptic(self, pattern=None, identifier=None):
        """Send haptic feedback. Default: short double-tap."""
        if pattern is None:
            pattern = [200, 100, 200]
        if identifier is None and self.connected_devices:
            identifier = list(self.connected_devices.keys())[0]
        if identifier:
            self.sdk.send_vibration_sequence(pattern, identifier)

    def run(self):
        """Block and run the SDK event loop."""
        self.connect()
        print("[TAPS] Waiting for Tap Strap 2 connection...")
        print("[TAPS] Make sure device is paired in Windows Bluetooth Settings")
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[TAPS] Stopping...")
            self._running = False

    def stop(self):
        self._running = False
