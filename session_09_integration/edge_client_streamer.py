import socket
import struct
import pickle
import time
import cv2
import pyrealsense2 as rs

def run_edge_client(server_ip="10.10.9.19", port=9999):
    print(f"[*] Connecting UP Square 6000 Edge to HPC Server at {server_ip}:{port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    print("[+] Connected to HPC Server!")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    payload_size = struct.calcsize("Q")

    try:
        while True:
            t_start = time.time()
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = color_frame.get_data()
            serialized = pickle.dumps(frame)

            client_socket.sendall(struct.pack("Q", len(serialized)) + serialized)

            data = b""
            while len(data) < payload_size:
                data += client_socket.recv(4096 * 8)
            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_size)[0]

            while len(data) < msg_size:
                data += client_socket.recv(4096 * 8)

            resp_frame = pickle.loads(data[:msg_size])
            total_roundtrip_ms = (time.time() - t_start) * 1000.0

            cv2.putText(resp_frame, f"Total Roundtrip (Network+HPC): {total_roundtrip_ms:.1f} ms",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("UP Square 6000 -> HPC Remote Inference View", resp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    ip = sys.argv[1] if len(sys.argv) > 1 else "10.10.9.19"
    run_edge_client(server_ip=ip)
