import time
import numpy as np
import openvino as ov

def benchmark_inference(model_xml_path, device_name="CPU", iterations=100):
    core = ov.Core()
    print(f"[*] Compiling model for device: {device_name}...")
    try:
        model = core.read_model(model_xml_path)
        compiled_model = core.compile_model(model, device_name)
    except Exception as e:
        print(f"[!] Device {device_name} unavailable: {e}")
        return

    infer_request = compiled_model.create_infer_request()
    input_tensor = np.random.randn(1, 3, 640, 640).astype(np.float32)

    for _ in range(10):
        infer_request.infer({0: input_tensor})

    start_time = time.time()
    for _ in range(iterations):
        infer_request.infer({0: input_tensor})
    total_time = time.time() - start_time

    avg_latency_ms = (total_time / iterations) * 1000.0
    fps = iterations / total_time
    print(f"--> [{device_name}] Latency: {avg_latency_ms:.2f} ms | Throughput: {fps:.2f} FPS")

if __name__ == "__main__":
    import sys
    model_xml = sys.argv[1] if len(sys.argv) > 1 else "models/best_openvino_model/best.xml"
    
    core = ov.Core()
    for dev in core.available_devices:
        benchmark_inference(model_xml, dev)
