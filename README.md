本项目是原来项目：https://github.com/ikun5200/siliconflow-api 的分支，其支持了openrouterAPI 代理

# openrouter API Proxy with docker
本项目是一个基于 Flask 构建的 openrouter API 代理服务器，支持多种大语言模型的调用，包括 Claude、GPT-4、Gemma、Mixtral 等。

# 本项目是一个基于 Flask 构建的 openrouter API 代理，具有以下功能：

-   **API 密钥轮询:**  支持多个 API 密钥轮询调用，自动处理速率限制和错误。
-   **模型管理:**  自动刷新模型列表，并区分免费和付费模型。
-   **实时监控:**  提供一个简约的仪表盘，实时监控请求速率 (RPM, RPD) 和 Token 使用量 (TPM, TPD)。
-   **密钥余额:**  显示 API 密钥的余额信息 (密钥部分打码以保护隐私)。
-   **响应式设计:**  仪表盘适配手机端和电脑端。


**Demo:**

![项目 Demo 截图](https://img.xwyue.com/i/2025/02/05/67a30fd64336d.png)

## 技术栈

-   **后端:** Python, Flask
-   **前端:** HTML, CSS, Bootstrap 5
-   **部署:** HUggingface

## 安装和部署步骤

### 1. 准备 API 密钥

在环境变量 `KEYS` 中设置你的 Siliconflow API 密钥，多个密钥用逗号分隔。例如：

```
KEYS=sk-key1,sk-key2,sk-key3
```

可选地，你还可以设置 `AUTHORIZATION_KEY` 环境变量用于访问仪表盘的身份验证。

### 2. 一键部署到 huggingface
https://huggingface.co/spaces/hf-demo-linux/sili?duplicate=true




## 配置说明

### 环境变量

| 变量名             | 必填 | 说明                                                                                                                                                              |
| :----------------- | :--- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KEYS`             | 是   | 你的 Siliconflow API 密钥，多个密钥用逗号分隔。                                                                                                                 |
| `AUTHORIZATION_KEY` | 否   | 用于请求API的身份验证密钥。                                                                                   |
| `TEST_MODEL`       | 否   | 用于测试 API 密钥是否有效的模型名称。 默认为 `Pro/google/gemma-2-9b-it`。                                                                                           |
| `BAN_MODELS`       | 否   | 一个 JSON 数组，包含要禁用的模型名称。 例如 `["model1", "model2"]`。                                                                                                |
| `PORT`             | 否   | 应用监听的端口号。 Vercel 会自动设置此变量，通常不需要手动设置。 默认值为 `7860`。                                                                                      |
| `FREE_MODEL_TEST_KEY` | 否 | 用于测试模型的可用性的免费API密钥。如果你没有提供自己的，则默认使用`FREE_MODEL_TEST_KEY`中提供的密钥。这通常是一个可以用来测试模型可用性，但没有太多额度的密钥。 |
| `API_PREFIX` | 否 | API路由的前缀。例如设置为 `/api` 则所有API路由都会加上 `/api` 前缀。默认为空。 |



## 使用示例

部署完成后，你可以通过 huggingface 提供的域名访问你的 API 代理。

-   **仪表盘:**  访问你的 Huggingface Space 的根 URL，即可查看仪表盘。
-   使用时请在客户端请求url结尾加上/handsome


## 许可证

本项目采用 MIT 许可证。 请参阅 [LICENSE](LICENSE) 文件了解更多信息。

---
