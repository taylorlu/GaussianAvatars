import asyncio
import websockets
import numpy as np
import time

flame_param = np.load(r'E:\workplace\AvatarSplat\GaussianAvatars\output\flame_param.npz')

async def send_file(websocket, path):
    fps = 30

    try:
        buffer_length = 168
        float32_array = np.zeros(buffer_length, dtype=np.float32)  # Example data
        i = 0
        # Step 3: Send the buffer over the WebSocket connection
        while(True):
            float32_array = np.concatenate([np.zeros([100]), flame_param['expr'][i, :50], np.zeros([3]), flame_param['neck_pose'][i], flame_param['jaw_pose'][i], flame_param['eyes_pose'][i], flame_param['translation'][i]], dtype=np.float32)
            print(i)
            await websocket.send(float32_array.tobytes())
            await asyncio.sleep(0.033)
            i += 1
            if(i>72):
                i = 0

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(send_file, "0.0.0.0", 8001):
        print("Server started on ws://0.0.0.0:8001")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())