# ClearVision RealSense Client
import pyrealsense2 as rs
import numpy as np
import cv2
import requests
import io
import time
import threading
import json
import os
from BLE import BLEWrapper
from asyncio.streams import logger
from raspi.modules.elevenlabs import ElevenLabs
from raspi.modules.gemini_vision import GeminiVision
# ClearVision RealSense Client
import asyncio
from sys import modules
from elevenlabs.play import play
from asyncio.streams import logger
import pyrealsense2 as rs
import numpy as np
import cv2
import requests
import io
import time
import subprocess
import threading
import json
import os


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Get from environment variable
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')


# HTTP endpoint configuration
API_ENDPOINT = "http://ec2-44-242-141-44.us-west-2.compute.amazonaws.com:80/upload"  # Change this to your actual endpoint

# BLE configuration
BLE_DEVICE_NAME = "FeatherBLE-Haptics"

# Initialize BLE device
ble_device = None
try:
    ble_device = BLEWrapper(device_name=BLE_DEVICE_NAME)
    ble_device.connect()
    print("BLE device connected successfully")
except Exception as e:
    print(f"Error connecting to BLE device: {e}")
    ble_device = None
    

gemini_vision = GeminiVision(GEMINI_API_KEY)
elevenlabs = ElevenLabs(ELEVENLABS_API_KEY)

def send_buzz_values_to_ble(buzz_values):
    """Send buzz values to BLE device using the BLE class"""
    global ble_device
    
    if ble_device is None:
        print("BLE device not available")
        return
    
    try:
        # Convert buzz values (0-1 range) to 0-10 range for BLE
        a = int(round(buzz_values[0] * 10)) if len(buzz_values) > 0 else 0
        b = int(round(buzz_values[1] * 10)) if len(buzz_values) > 1 else 0  
        c = int(round(buzz_values[2] * 10)) if len(buzz_values) > 2 else 0
        
        # Clamp values to 0-10 range
        a = max(0, min(10, a))
        b = max(0, min(10, b))
        c = max(0, min(10, c))
        
        print(f"Sending buzz values to BLE: {a} {b} {c}")
        
        # Send buzz values directly to BLE device
        ble_device.buzz([a/10.0, b/10.0, c/10.0])  # Convert back to 0-1 range for BLE class
        
    except Exception as e:
        print(f"Error sending buzz values to BLE: {e}")
        


def call_vlm_async(color_bytes: bytes, g: GeminiVision, e: ElevenLabs):
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run the async Gemini call
        description = loop.run_until_complete(
            g.describe_image_async(color_bytes)
        )

        description = json.loads(description)
        
        
        print(description)

        if description:
            logger.info(f"Gemini description received: {str(description)[:200]}...")
            
            # Generate and play audio
            if description.get('type') == 'alert':
                audio = loop.run_until_complete(
                    e.generate_speech(description.get('msg'))
                )
            if audio:
                try:
                    play(audio)
                    logger.info("Audio played successfully")
                except Exception as e:
                    logger.error(f"Error playing audio: {e}")
            else:
                logger.warning("No audio generated")

        else:
            logger.warning("No description received from Gemini")
    except Exception as e:
        logger.error(f"Error in async Gemini call: {e}")
    finally:
        print("Closing loop")
        loop.close()

try:
    align_to = rs.stream.color
    align = rs.align(align_to)

    i = 0
    while True:  # Run forever
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())

        h, w = depth_image.shape
        cx, cy = w // 2, h // 2
        center_depth_m = depth.get_distance(cx, cy)

        print(f"[{i:02d}] color={color_image.shape}, depth={depth_image.shape}, center_depth={center_depth_m:.3f} m")
        
        # Send HTTP request with images using ClearVision format
        try:
            # Encode color image to JPEG
            _, color_buffer = cv2.imencode('.jpg', color_image)
            color_bytes = color_buffer.tobytes()
            
            # Prepare depth data as JPEG image
            # Normalize depth to 0-255 range for JPEG
            depth_normalized = cv2.convertScaleAbs(depth_image, alpha=255.0/6000.0)  # Scale to 0-255, max 10m
            _, depth_buffer = cv2.imencode('.jpg', depth_normalized)
            depth_bytes = depth_buffer.tobytes()
            depth_size = len(depth_bytes)
            
            print(f"Depth data size: {depth_size:,} bytes ({depth_size/1024:.1f} KB)")
            
            if (i % 2 == 0):
                call_vlm_async(color_bytes, gemini_vision, elevenlabs)
            
            # Create files dictionary for multipart form data
            files = {
                'image': ('realsense_frame.jpg', io.BytesIO(color_bytes), 'image/jpeg'),
                'depth': ('depth_data.jpg', io.BytesIO(depth_bytes), 'image/jpeg')
            }
            
            # Additional data
            data = {
                'frame_number': str(i),
                'center_depth_m': str(center_depth_m),
                'timestamp': str(time.time()),
                'camera_type': 'realsense',
                'image_shape': f"{color_image.shape[1]}x{color_image.shape[0]}",
                'depth_shape': f"{depth_image.shape[1]}x{depth_image.shape[0]}"
            }
            
            # Send HTTP request using multipart form data
            response = requests.post(API_ENDPOINT, files=files, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"HTTP Response for frame {i}: {result}")
                
                # Extract buzz values from response
                try:
                    buzz_values = result.get('data', {}).get('vision_processing', {}).get('depth_processing', {}).get('buzz_values', [])
                    if buzz_values:
                        print(f"Extracted buzz values: {buzz_values}")
                        send_buzz_values_to_ble(buzz_values)
                    else:
                        print("No buzz values found in response")
                except Exception as e:
                    print(f"Error extracting buzz values: {e}")
            else:
                print(f"HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed for frame {i}: {e}")
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
        
        i += 1  # Increment frame counter
finally:
    # Clean up BLE device
    if ble_device is not None:
        try:
            ble_device.stop_buzz()  # Stop all buzzers
            print(f"Stopped buzzers")
            print(f"Disconnecting BLE device")
            ble_device.disconnect()
            print(f"Disconnected BLE device")
        except Exception as e:
            print(f"Error disconnecting BLE device: {e}")