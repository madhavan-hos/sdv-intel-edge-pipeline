# Intel Unnati HPC & Edge AI Vision Stack: SDV Workshop
This repository contains the complete session-by-session codebase for the Build Your First SDV Vehicles workshop.

## Directory Structure
sdv_workshop_project/
|-- session_03_setup/
|-- session_04_vision/
|-- session_05_model_hpc/
|-- session_06_fusion/
|-- session_07_acceleration/
|-- session_08_usecase/
|-- session_09_integration/
|-- models/
    |-- best_openvino_model/

## Quick Start
1. conda activate ai
2. pip install -r session_03_setup/requirements.txt
3. python session_09_integration/local_sdv_pipeline.py models/best_openvino_model/ 0
