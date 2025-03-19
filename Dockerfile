# 使用官方 Python 运行时作为父镜像
FROM python:3.12-rc-slim-buster

# 设置工作目录为 /app
WORKDIR /app

# 将当前目录下的 requirements.txt 复制到容器的 /app/requirements.txt 中
COPY requirements.txt /app/

# 安装 requirements.txt 中指定的任何所需软件包
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录下的其余文件复制到容器的 /app 中
COPY . /app/

# 使端口 7860 可供此容器外的世界使用
EXPOSE 7860

# 设置环境变量 PORT，以适应 Hugging Face Space 的要求
ENV PORT=7860

# 在容器启动时运行 app.py
CMD ["python", "app.py"]
