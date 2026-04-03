from .vision_process import (
    extract_vision_info,
    fetch_image,
    fetch_video,
    process_vision_info,
    smart_resize,
)

import os

if '__printed_qwen_vl_utils_loading_message' not in globals():
    print("Hello! You are loading qwen_vl_utils from:", os.path.abspath(__file__))
    __printed_qwen_vl_utils_loading_message = True