# Siliconflow API Proxy with Vercel Dashboard

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fikun5200%2Fsiliconflow-api&env=KEYS,AUTHORIZATION_KEY&project-name=siliconflow-api&repo-name=siliconflow-api)

本项目是一个基于 Flask 构建的 Siliconflow API 代理，具有以下功能：

-   **API 密钥轮询:**  支持多个 API 密钥轮询调用，自动处理速率限制和错误。
-   **模型管理:**  自动刷新模型列表，并区分免费和付费模型。
-   **实时监控:**  提供一个简约的仪表盘，实时监控请求速率 (RPM, RPD) 和 Token 使用量 (TPM, TPD)。
-   **密钥余额:**  显示 API 密钥的余额信息 (密钥部分打码以保护隐私)。
-   **Vercel 部署:**  专为 Vercel 平台优化，可以一键部署。
-   **响应式设计:**  仪表盘适配手机端和电脑端。

## 功能特性

-   代理 Siliconflow API 的以下端点：
    -   `/v1/chat/completions` (聊天)
    -   `/v1/embeddings` (嵌入)
    -   `/v1/images/generations` (图像生成)
    -   `/v1/models` (模型列表)
    -   `/v1/dashboard/billing/usage` (账单使用情况)
    -   `/v1/dashboard/billing/subscription` (账单订阅信息)
-   支持多 API 密钥轮询，自动处理速率限制和错误。
-   自动刷新模型列表，并区分免费和付费模型。
-   提供一个简约的仪表盘，实时监控请求速率和 Token 使用量。
-   显示 API 密钥的余额信息 (密钥部分打码)。
-   支持 Vercel 一键部署。

**Demo:**

![项目 Demo 截图](https://img.xwyue.com/i/2025/02/05/67a30fd64336d.png)

## 技术栈

-   **后端:** Python, Flask
-   **前端:** HTML, CSS, Bootstrap 5
-   **部署:** Vercel

## 安装和部署步骤

### 1. 准备 API 密钥

在环境变量 `KEYS` 中设置你的 Siliconflow API 密钥，多个密钥用逗号分隔。例如：

```
KEYS=sk-key1,sk-key2,sk-key3
```

可选地，你还可以设置 `AUTHORIZATION_KEY` 环境变量用于访问仪表盘的身份验证。

### 2. 一键部署到 Vercel

点击下面的按钮，一键部署到 Vercel：

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fikun5200%2Fsiliconflow-api&env=KEYS,AUTHORIZATION_KEY&project-name=siliconflow-api&repo-name=siliconflow-api)

在部署过程中，Vercel 会提示你输入环境变量 `KEYS` 和 `AUTHORIZATION_KEY` 的值。

### 3. 手动部署 (可选)

如果你想手动部署，请按照以下步骤操作：

1. 将本项目克隆到本地：

    ```bash
    git clone https://github.com/ikun5200/siliconflow-api.git
    cd siliconflow-api
    ```

2. 安装依赖：

    ```bash
    pip install -r requirements.txt
    ```

3. 在本地运行 (可选，用于测试)：

    ```bash
    python app.py
    ```

4. 将代码推送到你的 GitHub 仓库。

5. 在 Vercel 网站上创建一个新项目，并导入你的 GitHub 仓库。

6. 在 Vercel 项目的设置中，配置环境变量 `KEYS` 和 `AUTHORIZATION_KEY`。

7. 部署你的应用。

## 配置说明

### 环境变量

| 变量名             | 必填 | 说明                                                                                                                                                              |
| :----------------- | :--- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KEYS`             | 是   | 你的 Siliconflow API 密钥，多个密钥用逗号分隔。                                                                                                                 |
| `AUTHORIZATION_KEY` | 否   | 用于请求API的身份验证密钥。                                                                               |
| `TEST_MODEL`       | 否   | 用于测试 API 密钥是否有效的模型名称。 默认为 `Pro/google/gemma-2-9b-it`。                                                                                           |
| `BAN_MODELS`       | 否   | 一个 JSON 数组，包含要禁用的模型名称。 例如 `["model1", "model2"]`。                                                                                                |
| `PORT`             | 否   | 应用监听的端口号。 Vercel 会自动设置此变量，通常不需要手动设置。 默认值为 `7860`。                                                                                      |
| `FREE_MODEL_TEST_KEY` | 否 | 用于测试模型的可用性的免费API密钥。如果你没有提供自己的，则默认使用`FREE_MODEL_TEST_KEY`中提供的密钥。这通常是一个可以用来测试模型可用性，但没有太多额度的密钥。 |

### `vercel.json`

你可以在 `vercel.json` 文件中配置 Vercel 的构建和路由行为。

-   **`maxDuration`:**  设置 Serverless Function 的最大执行时间 (以秒为单位)。 默认值为 10 (Hobby Plan)，最大值为 60 (Pro Plan)。 如果你的请求处理时间超过了限制，请根据你的套餐计划调整此值。

## 使用示例

部署完成后，你可以通过 Vercel 提供的域名访问你的 API 代理。

-   **仪表盘:**  访问你的 Vercel 应用的根 URL (例如 `https://siliconflow-api.vercel.app/`)，即可查看仪表盘。
-   **API 代理:**  将你的 Siliconflow API 请求发送到你的 Vercel 应用的相应路径 (例如 `https://siliconflow-api.vercel.app/v1/chat/completions`)，即可通过你的代理访问 Siliconflow API。


## 许可证

本项目采用 MIT 许可证。 请参阅 [LICENSE](LICENSE) 文件了解更多信息。

---
