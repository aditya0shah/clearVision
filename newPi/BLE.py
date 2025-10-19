import asyncio
import time
from typing import Optional
from bleak import BleakClient, BleakScanner

# Nordic UART Service UUIDs
SERVICE_UUID = '6E400001-B5A3-F393-E0A9-E50E24DCCA9E'
RX_UUID      = '6E400002-B5A3-F393-E0A9-E50E24DCCA9E'  # Write
TX_UUID      = '6E400003-B5A3-F393-E0A9-E50E24DCCA9E'  # Notify

def log(s: str) -> None:
    print(s, flush=True)

class BLE:
    def __init__(self, device_name: str = "FeatherBLE-Haptics", device_address: Optional[str] = None):
        self.device_name = device_name
        self.device_address = device_address
        self.client = None
        self.addr = None
        
    async def find_device(self, timeout: float = 10.0):
        """Find the BLE device by name or use provided address"""
        if self.device_address:
            self.addr = self.device_address
            return self.addr
            
        log(f"Scanning for BLE devices for up to {timeout:.0f}s...")
        dev = await BleakScanner.find_device_by_filter(
            lambda d, ad: (d.name == self.device_name) if self.device_name else True,
            timeout=timeout
        )
        if not dev:
            raise RuntimeError("Device not found. Try --name FeatherBLE-Haptics or --address <mac/uuid>.")
        log(f"Found device: name={dev.name} address={dev.address}")
        self.addr = dev.address
        return self.addr

    async def connect(self):
        """Connect to the BLE device"""
        if not self.addr:
            await self.find_device()
            
        self.client = BleakClient(self.addr)
        await self.client.connect()
        
        if not self.client.is_connected:
            raise RuntimeError("Failed to connect to BLE device")
        
        log("Connected to BLE device")
        return True

    async def disconnect(self):
        """Disconnect from the BLE device"""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            log("Disconnected from BLE device")

    async def buzz(self, buzz_values):
        """Send buzz values to the BLE device"""
        if not self.client or not self.client.is_connected:
            raise RuntimeError("BLE device not connected. Call connect() first.")
            
        a = int(round(buzz_values[0] * 10)) if len(buzz_values) > 0 else 0
        b = int(round(buzz_values[1] * 10)) if len(buzz_values) > 1 else 0  
        c = int(round(buzz_values[2] * 10)) if len(buzz_values) > 2 else 0
        
        # Clamp values to 0-10 range
        a = max(0, min(10, a))
        # b = max(0, min(10, b))
        b = 0
        c = max(0, min(10, c))
        
        message = f"{a} {b} {c}\n".encode("utf-8")
        await self.client.write_gatt_char(RX_UUID, message, response=False)
        log(f"Sent buzz values: {a} {b} {c}")

    async def stop_buzz(self):
        """Stop all buzzers by sending 0 0 0"""
        if not self.client or not self.client.is_connected:
            raise RuntimeError("BLE device not connected. Call connect() first.")
            
        await self.client.write_gatt_char(RX_UUID, b"0 0 0\n", response=False)
        log("Stopped all buzzers")
        

# Example usage function
async def example_usage():
    """Example of how to use the BLE class"""
    ble = BLE()
    
    try:
        # Connect to the device
        await ble.connect()
        
        # Send some buzz values (0-1 range, will be converted to 0-10)
        await ble.buzz([0.5, 1.0, 0.3])  # 50%, 100%, 30% intensity
        await asyncio.sleep(1)  # Wait 1 second
        
        # Stop buzzing
        await ble.stop_buzz()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always disconnect
        await ble.disconnect()

class BLEWrapper:
    """Synchronous wrapper for the BLE class for easier use in non-async contexts"""
    
    def __init__(self, device_name: str = "FeatherBLE-Haptics", device_address: Optional[str] = None):
        self.ble = BLE(device_name, device_address)
        self._loop = None
        
    def _get_loop(self):
        """Get or create event loop"""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop
    
    def connect(self):
        """Synchronous connect"""
        loop = self._get_loop()
        return loop.run_until_complete(self.ble.connect())
    
    def disconnect(self):
        """Synchronous disconnect"""
        loop = self._get_loop()
        return loop.run_until_complete(self.ble.disconnect())
    
    def buzz(self, buzz_values):
        """Synchronous buzz"""
        loop = self._get_loop()
        return loop.run_until_complete(self.ble.buzz(buzz_values))
    
    def stop_buzz(self):
        """Synchronous stop buzz"""
        loop = self._get_loop()
        return loop.run_until_complete(self.ble.stop_buzz())