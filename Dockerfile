FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 安装 curl (健康检查需要)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 可选：切换国内源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖
RUN pip install --no-cache-dir \
    transformers==4.57.4 \
    funasr==1.3.1 \
    openai-whisper \
    flask

# 克隆 Fun-ASR 源码（部分模型可能需要）
RUN git clone https://github.com/FunAudioLLM/Fun-ASR.git /app/Fun-ASR && \
    cd /app/Fun-ASR && \
    pip install --no-cache-dir -e .

COPY api_funasr.py /app/

EXPOSE 5045

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5045/health || exit 1

CMD ["python", "api_funasr.py"]
