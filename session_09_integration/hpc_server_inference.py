import socket
import struct
import pickle
import cv2
import numpy as np
import time
from ultralytics import YOLO

def start_server(host="0.0.0.0", port=9999, model_path="models/best_openvino_model/"):
    print(f"[*] Initializing OpenVINO Model on HPC Server: {model_path}")
    model = YOLO(model_path, task="detect")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[*] HPC Server listening on {host}:{port} (Active IP: 10.10.9.19)")

    while True:
        conn, addr = server_socket.accept()
        print(f"[+] Edge Device Connected from: {addr}")
        data = b""
        payload_size = struct.calcsize("Q")

        try:
            while True:
                while len(data) < payload_size:
                    packet = conn.recv(4096 * 8)
                    if not packet:
                        break
                    data += packet
                if not data:
                    break

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += conn.recv(4096 * 8)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame = pickle.loads(frame_data)
                
                t0 = time.time()
                results = model.predict(source=frame, imgsz=640, device="intel:cpu", verbose=False)
                latency_ms = (time.time() - t0) * 1000.0
                
                annotated = results[0].plot()
                cv2.putText(annotated, f"HPC Inference Latency: {latency_ms:.1f} ms", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                resp_data = pickle.dumps(annotated)
                conn.sendall(struct.pack("Q", len(resp_data)) + resp_data)
        except ConnectionResetError:
            print(f"[-] Edge Device {addr} disconnected.")
        finally:
            conn.close()

if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/best_openvino_model/"
    start_server(model_path=model_dir)
