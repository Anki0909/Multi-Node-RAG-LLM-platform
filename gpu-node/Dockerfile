FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git cmake build-essential \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="50;52;60;61;70;75;80;86;89;90" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64 -lcuda" \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64 -lcuda" && \
    cmake --build build -j$(nproc)

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries from builder
COPY --from=builder /app/llama.cpp/build/bin/* /usr/local/bin/

# Create symlink for common binary name
RUN ln -sf /usr/local/bin/llama-cli /usr/local/bin/llama || true && \
    chmod +x /usr/local/bin/llama* 2>/dev/null || true

# Copy all CUDA and llama.cpp shared libraries from builder
# COPY --from=builder /usr/local/cuda/lib64/libcuda.so* /usr/local/cuda/lib64/
COPY --from=builder /app/llama.cpp/build/bin/*.so* /usr/local/lib/
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/llama.conf && \
    ldconfig
RUN ldconfig

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV MODEL_PATH=/mnt/data/rag/models
ENV CACHE_PATH=/mnt/data/rag/cache

EXPOSE 8000

CMD ["llama-server", \
  "--model", "${MODEL_PATH}/qwen2.5-3b-instruct-q4_k_m.gguf", \
  "--ctx-size", "4096", \
  "--n-gpu-layers", "28", \
  "--host", "0.0.0.0", \
  "--port", "8000"]
