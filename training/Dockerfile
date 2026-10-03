# EquiCave training image (CUDA). Build: docker build -t equicave-train -f training/Dockerfile .
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /work
COPY pyproject.toml ./
RUN pip install --no-cache-dir numpy scipy pandas scikit-learn lightgbm pyyaml pytest transformers safetensors tqdm
COPY . .
ENV PYTHONPATH=/work/src:/work
CMD ["python", "-m", "training", "list"]
