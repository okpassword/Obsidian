import os
import sys
import time
import base64
import logging
import traceback
import torch
import tempfile
from flask import Flask, request, jsonify
from funasr import AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FunASR_API")

MODEL_DIR = "FunAudioLLM/Fun-ASR-Nano-2512"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

logger.info(f"使用设备: {DEVICE}")

try:
    model = AutoModel(model=MODEL_DIR, device=DEVICE)
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    sys.exit(1)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

@app.route('/whisper', methods=['POST'])
def asr_api():
    start_time = time.time()
    temp_file_path = None

    try:
        data = request.get_json()
        if not data or 'base64' not in data:
            return jsonify({"status": 400, "text": "缺少 base64 字段"}), 400

        base64_str = data['base64']
        if ',' in base64_str:
            base64_str = base64_str.split(',')[-1]

        audio_bytes = base64.b64decode(base64_str)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_file_path = tmp.name

        res = model.generate(input=temp_file_path, batch_size=1, itn=True)

        text = res[0].get("text", "") if res else ""

        logger.info(f"识别完成: {text} (耗时 {time.time()-start_time:.2f}s)")
        return jsonify({"status": 200, "text": text})

    except Exception as e:
        logger.error(f"推理失败: {e}")
        traceback.print_exc()
        return jsonify({"status": 500, "text": str(e)}), 500

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")
        torch_gc()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5045, threaded=False)
