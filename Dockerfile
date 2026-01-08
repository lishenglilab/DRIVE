# ==============================================================================
# Dockerfile for DRP Unified Workflow (v2.0 - Final Dockerized Version)
# ==============================================================================
FROM docker.m.daocloud.io/library/ubuntu:22.04

# 环境变量设置
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    CUDA_VERSION_SHORT=11.8 \
    # 核心：告诉 Nextflow 去哪找已经装好的环境，且严禁联网
    NXF_CONDA_CACHEDIR=/opt/conda/envs \
    NXF_OFFLINE=true

# 1. 系统依赖 & CUDA (使用阿里云源)
RUN sed -i 's#http://archive.ubuntu.com/ubuntu/#http://mirrors.aliyun.com/ubuntu/#' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gnupg wget procps software-properties-common && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && apt-get -y install cuda-toolkit-11-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${CONDA_DIR}/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

# 2. 安装 Miniconda
RUN wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh

# 3. 配置源并安装 Mamba & Nextflow
# 【核心优化】：大幅增加超时时间，防止 4GB 包下载失败
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    conda config --set show_channel_urls yes && \
    conda config --set remote_read_timeout_secs 1200.0 && \
    conda config --set remote_connect_timeout_secs 1200.0 && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    conda install -c conda-forge mamba -y && \
    mamba install -c bioconda nextflow -y && \
    nextflow -version && \
    conda clean -afy

# 4. 复制项目文件
WORKDIR /app
COPY . .

# 5. 【核心预装】在构建时安装环境及 local_wheels (带重试逻辑)
RUN echo "Building part1_env..." && \
    (mamba env create -y -n part1_env -f environments/part1_env.yml || \
     mamba env create -y -n part1_env -f environments/part1_env.yml) && \
    /opt/conda/envs/part1_env/bin/pip install ./local_wheels/cp37/*.whl --no-deps && \
    /opt/conda/envs/part1_env/bin/pip install torch-geometric==2.0.3 && \
    \
    echo "Building dipk_graphdrp_env..." && \
    (mamba env create -y -n dipk_graphdrp_env -f environments/dipk_graphdrp_env.yml || \
     mamba env create -y -n dipk_graphdrp_env -f environments/dipk_graphdrp_env.yml) && \
    /opt/conda/envs/dipk_graphdrp_env/bin/pip install ./local_wheels/cp38/*.whl --no-deps && \
    /opt/conda/envs/dipk_graphdrp_env/bin/pip install torch-geometric==2.0.3 && \
    conda clean -afy

# 6. 清理临时文件并赋权
RUN rm -rf ./local_wheels ./environments && \
    chmod +x run_ensemble_dockerfile.sh

# 设置最终入口
ENTRYPOINT ["./run_ensemble_dockerfile.sh"]
