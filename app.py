import os, time, logging, requests, json, uuid, concurrent.futures, threading, base64, io
from io import BytesIO
from itertools import chain
from PIL import Image
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from werkzeug.middleware.proxy_fix import ProxyFix
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# 设置时区
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API 端点
API_ENDPOINT = "https://api-st.siliconflow.cn/v1/user/info"
TEST_MODEL_ENDPOINT = "https://api-st.siliconflow.cn/v1/chat/completions"
MODELS_ENDPOINT = "https://api-st.siliconflow.cn/v1/models"
EMBEDDINGS_ENDPOINT = "https://api-st.siliconflow.cn/v1/embeddings"
IMAGE_ENDPOINT = "https://api-st.siliconflow.cn/v1/images/generations"

# 设置请求会话，包括重试机制
def requests_session_with_retries(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=1000,
        pool_maxsize=10000,
        pool_block=False
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = requests_session_with_retries()

# 初始化 Flask 应用
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

# 模型和密钥状态的全局变量
models = {
    "text": [],
    "free_text": [],
    "embedding": [],
    "free_embedding": [],
    "image": [],
    "free_image": []
}
key_status = {
    "invalid": [],
    "free": [],
    "unverified": [],
    "valid": []
}

# 用于并发执行任务的线程池
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10000)

# 模型密钥索引，用于轮询选择密钥
model_key_indices = {}

# 请求时间戳和令牌计数，用于简单的速率监控
request_timestamps = []
token_counts = []
request_timestamps_day = []
token_counts_day = []

# 数据锁，用于保护共享数据结构的并发访问
data_lock = threading.Lock()

# 获取信用摘要的函数
def get_credit_summary(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = session.get(API_ENDPOINT, headers=headers, timeout=2)  # 设置了 2 秒超时
            response.raise_for_status()
            data = response.json().get("data", {})
            total_balance = data.get("totalBalance", 0)
            logging.info(f"获取额度，API Key：{api_key}，当前额度: {total_balance}")
            return {"total_balance": float(total_balance)}
        except requests.exceptions.Timeout as e:
            logging.error(
                f"获取额度信息失败，API Key：{api_key}，尝试次数：{attempt + 1}/{max_retries}，错误信息：{e} (Timeout)")
            if attempt >= max_retries - 1:
                logging.error(f"获取额度信息失败，API Key：{api_key}，所有重试次数均已失败 (Timeout)")
        except requests.exceptions.RequestException as e:
            logging.error(f"获取额度信息失败，API Key：{api_key}，错误信息：{e}")
            return None

# 用于模型可用性测试的免费模型测试密钥和免费图像列表
FREE_MODEL_TEST_KEY = (
    "sk-bmjbjzleaqfgtqfzmcnsbagxrlohriadnxqrzfocbizaxukw"  # 你需要替换为有效的测试密钥
)
FREE_IMAGE_LIST = [
    "stabilityai/stable-diffusion-3-5-large",
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-3-medium",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2-1"
]

# 测试模型可用性的函数
def test_model_availability(api_key, model_name, model_type="chat"):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if model_type == "image":
        return model_name in FREE_IMAGE_LIST
    try:
        endpoint = EMBEDDINGS_ENDPOINT if model_type == "embedding" else TEST_MODEL_ENDPOINT
        payload = (
            {"model": model_name, "input": ["hi"]}
            if model_type == "embedding"
            else {"model": model_name, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5,
                  "stream": False}
        )
        timeout = 5 if model_type == "embedding" else 2  # 减少了超时时间
        response = session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        return response.status_code in [200, 429]
    except requests.exceptions.RequestException as e:
        logging.error(
            f"测试{model_type}模型 {model_name} 可用性失败，"
            f"API Key：{api_key}，错误信息：{e}"
        )
        return False

# 处理图像 URL 的函数
def process_image_url(image_url, response_format=None):
    if not image_url:
        return {"url": ""}
    if response_format == "b64_json":
        try:
            response = session.get(image_url, stream=True, timeout=2)  # 设置了 2 秒超时
            response.raise_for_status()
            image = Image.open(response.raw)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {"b64_json": img_str}
        except Exception as e:
            logging.error(f"图片转base64失败: {e}")
            return {"url": image_url}
    return {"url": image_url}

# 创建 base64 编码的 Markdown 图像链接的函数
def create_base64_markdown_image(image_url):
    try:
        response = session.get(image_url, stream=True, timeout=2)  # 设置了 2 秒超时
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        new_size = tuple(dim // 4 for dim in image.size)
        resized_image = image.resize(new_size, Image.LANCZOS)
        buffered = BytesIO()
        resized_image.save(buffered, format="PNG")
        base64_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        markdown_image_link = f"![](data:image/png;base64,{base64_encoded})"
        logging.info("Created base64 markdown image link.")
        return markdown_image_link
    except Exception as e:
        logging.error(f"Error creating markdown image: {e}")
        return None

# 提取用户消息内容的函数
def extract_user_content(messages):
    user_content = ""
    for message in messages:
        if message["role"] == "user":
            if isinstance(message["content"], str):
                user_content += message["content"] + " "
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        user_content += item.get("text", "") + " "
    return user_content.strip()

# 获取 SiliconFlow 数据的函数
def get_siliconflow_data(model_name, data):
    siliconflow_data = {
        "model": model_name,
        "prompt": data.get("prompt") or "",
    }
    if model_name == "black-forest-labs/FLUX.1-pro":
        siliconflow_data.update({
            "width": max(256, min(1440, (data.get("width", 1024) // 32) * 32)),
            "height": max(256, min(1440, (data.get("height", 768) // 32) * 32)),
            "prompt_upsampling": data.get("prompt_upsampling", False),
            "image_prompt": data.get("image_prompt"),
            "steps": max(1, min(50, data.get("steps", 20))),
            "guidance": max(1.5, min(5, data.get("guidance", 3))),
            "safety_tolerance": max(0, min(6, data.get("safety_tolerance", 2))),
            "interval": max(1, min(4, data.get("interval", 2))),
            "output_format": data.get("output_format", "png")
        })
        seed = data.get("seed")
        if isinstance(seed, int) and 0 < seed < 9999999999:
            siliconflow_data["seed"] = seed
    else:
        siliconflow_data.update({
            "image_size": data.get("image_size", "1024x1024"),
            "prompt_enhancement": data.get("prompt_enhancement", False)
        })
        seed = data.get("seed")
        if isinstance(seed, int) and 0 < seed < 9999999999:
            siliconflow_data["seed"] = seed
        if model_name not in ["black-forest-labs/FLUX.1-schnell", "Pro/black-forest-labs/FLUX.1-schnell"]:
            siliconflow_data.update({
                "batch_size": max(1, min(4, data.get("n", 1))),
                "num_inference_steps": max(1, min(50, data.get("steps", 20))),
                "guidance_scale": max(0, min(100, data.get("guidance_scale", 7.5))),
                "negative_prompt": data.get("negative_prompt")
            })
    valid_sizes = ["1024x1024", "512x1024", "768x512", "768x1024", "1024x576", "576x1024", "960x1280", "720x1440",
                   "720x1280"]
    if "image_size" in siliconflow_data and siliconflow_data["image_size"] not in valid_sizes:
        siliconflow_data["image_size"] = "1024x1024"
    return siliconflow_data

# 刷新模型列表的函数
def refresh_models():
    global models

    # 使用线程池并发获取模型列表
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(get_all_models, FREE_MODEL_TEST_KEY, "chat"): "text",
            executor.submit(get_all_models, FREE_MODEL_TEST_KEY, "embedding"): "embedding",
            executor.submit(get_all_models, FREE_MODEL_TEST_KEY, "text-to-image"): "image"
        }

        for future in concurrent.futures.as_completed(futures):
            model_type = futures[future]
            try:
                models[model_type] = future.result()
            except Exception as exc:
                logging.error(f"获取 {model_type} 模型列表失败: {exc}")
                models[model_type] = []

    models["free_text"] = []
    models["free_embedding"] = []
    models["free_image"] = []
    ban_models = []
    ban_models_str = os.environ.get("BAN_MODELS")
    if ban_models_str:
        try:
            ban_models = json.loads(ban_models_str)
            if not isinstance(ban_models, list):
                logging.warning("环境变量 BAN_MODELS 格式不正确，应为 JSON 数组。")
                ban_models = []
        except json.JSONDecodeError:
            logging.warning("环境变量 BAN_MODELS JSON 解析失败，请检查格式。")
    models["text"] = [model for model in models["text"] if model not in ban_models]
    models["embedding"] = [model for model in models["embedding"] if model not in ban_models]
    models["image"] = [model for model in models["image"] if model not in ban_models]
    model_types = [
        ("text", "chat"),
        ("embedding", "embedding"),
        ("image", "image")
    ]
    for model_type, test_type in model_types:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10000) as executor:
            future_to_model = {
                executor.submit(
                    test_model_availability,
                    FREE_MODEL_TEST_KEY,
                    model,
                    test_type
                ): model for model in models[model_type]
            }
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    is_free = future.result()
                    if is_free:
                        models[f"free_{model_type}"].append(model)
                except Exception as exc:
                    logging.error(f"{model_type}模型 {model} 测试生成异常: {exc}")
    for model_type in ["text", "embedding", "image"]:
        logging.info(f"所有{model_type}模型列表：{models[model_type]}")
        logging.info(f"免费{model_type}模型列表：{models[f'free_{model_type}']}")

# 加载 API 密钥的函数
def load_keys():
    global key_status
    for status in key_status:
        key_status[status] = []
    keys_str = os.environ.get("KEYS")
    if not keys_str:
        logging.warning("环境变量 KEYS 未设置。")
        return
    test_model = os.environ.get("TEST_MODEL", "Pro/google/gemma-2-9b-it")
    unique_keys = list(set(key.strip() for key in keys_str.split(',')))
    os.environ["KEYS"] = ','.join(unique_keys)
    logging.info(f"加载的 keys：{unique_keys}")

    # 使用线程池并发处理密钥
    with concurrent.futures.ThreadPoolExecutor(max_workers=10000) as executor:
        futures = [executor.submit(process_key_with_logging, key, test_model) for key in unique_keys]
        concurrent.futures.wait(futures)

    for status, keys in key_status.items():
        logging.info(f"{status.capitalize()} KEYS: {keys}")

    global invalid_keys_global, free_keys_global, unverified_keys_global, valid_keys_global
    invalid_keys_global = key_status["invalid"]
    free_keys_global = key_status["free"]
    unverified_keys_global = key_status["unverified"]
    valid_keys_global = key_status["valid"]

# 对单个密钥进行处理的辅助函数
def process_key_with_logging(key, test_model):
    try:
        key_type = process_key(key, test_model)
        if key_type in key_status:
            key_status[key_type].append(key)
        return key_type
    except Exception as exc:
        logging.error(f"处理 KEY {key} 生成异常: {exc}")
        return "invalid"

# 处理 API 密钥的函数
def process_key(key, test_model):
    credit_summary = get_credit_summary(key)
    if credit_summary is None:
        return "invalid"
    else:
        total_balance = credit_summary.get("total_balance", 0)
        if total_balance <= 0.03:
            return "free"
        else:
            if test_model_availability(key, test_model):
                return "valid"
            else:
                return "unverified"

# 获取所有模型列表的函数
def get_all_models(api_key, sub_type):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = session.get(
            MODELS_ENDPOINT,
            headers=headers,
            params={"sub_type": sub_type}
        )
        response.raise_for_status()
        data = response.json()
        if (
                isinstance(data, dict) and
                'data' in data and
                isinstance(data['data'], list)
        ):
            return [
                model.get("id") for model in data["data"]
                if isinstance(model, dict) and "id" in model
            ]
        else:
            logging.error("获取模型列表失败：响应数据格式不正确")
            return []
    except requests.exceptions.RequestException as e:
        logging.error(
            f"获取模型列表失败，"
            f"API Key：{api_key}，错误信息：{e}"
        )
        return []
    except (KeyError, TypeError) as e:
        logging.error(
            f"解析模型列表失败，"
            f"API Key：{api_key}，错误信息：{e}"
        )
        return []

# 确定请求类型的函数
def determine_request_type(model_name, model_list, free_model_list):
    if model_name in free_model_list:
        return "free"
    elif model_name in model_list:
        return "paid"
    else:
        return "unknown"

# 选择 API 密钥的函数
def select_key(request_type, model_name):
    if request_type == "free":
        available_keys = (
                free_keys_global +
                unverified_keys_global +
                valid_keys_global
        )
    elif request_type == "paid":
        available_keys = unverified_keys_global + valid_keys_global
    else:
        available_keys = (
                free_keys_global +
                unverified_keys_global +
                valid_keys_global
        )
    if not available_keys:
        return None
    current_index = model_key_indices.get(model_name, 0)
    for _ in range(len(available_keys)):
        key = available_keys[current_index % len(available_keys)]
        current_index += 1
        if key_is_valid(key, request_type):
            model_key_indices[model_name] = current_index
            return key
        else:
            logging.warning(
                f"KEY {key} 无效或达到限制，尝试下一个 KEY"
            )
    model_key_indices[model_name] = 0
    return None

# 检查 API 密钥是否有效的函数
def key_is_valid(key, request_type):
    if request_type == "invalid":
        return False
    credit_summary = get_credit_summary(key)
    if credit_summary is None:
        return False
    total_balance = credit_summary.get("total_balance", 0)
    if request_type == "free":
        return True
    elif request_type == "paid" or request_type == "unverified":  # Fixed typo here
        return total_balance > 0
    else:
        return False

# 检查请求授权的函数
def check_authorization(request):
    authorization_key = os.environ.get("AUTHORIZATION_KEY")
    if not authorization_key:
        logging.warning("环境变量 AUTHORIZATION_KEY 未设置，此时无需鉴权即可使用，建议进行设置后再使用。")
        return True
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        logging.warning("请求头中缺少 Authorization 字段。")
        return False
    if auth_header != f"Bearer {authorization_key}":
        logging.warning(f"无效的 Authorization 密钥：{auth_header}")
        return False
    return True

# 对 API 密钥进行打码处理的函数
def obfuscate_key(key):
    if not key:
        return "****"
    prefix_length = 6
    suffix_length = 4
    if len(key) <= prefix_length + suffix_length:
        return "****"  # If key is too short, just mask it all
    prefix = key[:prefix_length]
    suffix = key[-suffix_length:]
    masked_part = "*" * (len(key) - prefix_length - suffix_length)
    return prefix + masked_part + suffix

# 设置后台调度器，用于定期加载密钥和刷新模型列表
scheduler = BackgroundScheduler()
scheduler.add_job(load_keys, 'interval', hours=1)
scheduler.remove_all_jobs()
scheduler.add_job(refresh_models, 'interval', hours=1)

# Flask 应用的路由定义
@app.route('/')
def index():
    current_time = time.time()
    one_minute_ago = current_time - 60
    one_day_ago = current_time - 86400
    with data_lock:
        while request_timestamps and request_timestamps[0] < one_minute_ago:
            request_timestamps.pop(0)
            token_counts.pop(0)
        rpm = len(request_timestamps)
        tpm = sum(token_counts)
    with data_lock:
        while request_timestamps_day and request_timestamps_day[0] < one_day_ago:
            request_timestamps_day.pop(0)
            token_counts_day.pop(0)
        rpd = len(request_timestamps_day)
        tpd = sum(token_counts_day)

    # 并发获取所有密钥的余额信息
    key_balances = []
    all_keys = list(chain(*key_status.values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10000) as executor:
        future_to_key = {executor.submit(get_credit_summary, key): key for key in all_keys}
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                credit_summary = future.result()
                balance = credit_summary.get("total_balance") if credit_summary else "获取失败"
                key_balances.append({"key": obfuscate_key(key), "balance": balance})
            except Exception as exc:
                logging.error(f"获取 KEY {obfuscate_key(key)} 余额信息失败: {exc}")
                key_balances.append({"key": obfuscate_key(key), "balance": "获取失败"})

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('index.html', rpm=rpm, tpm=tpm, rpd=rpd, tpd=tpd, key_balances=key_balances,
                           now=now)

# 去除了 /env_test 路由

# 列出所有可用模型的路由，去除了 /handsome 前缀
@app.route('/v1/models', methods=['GET'])
def list_models():
    if not check_authorization(request):
        return jsonify({"error": "Unauthorized"}), 401
    detailed_models = []
    all_models = chain(
        models["text"],
        models["embedding"],
        models["image"]
    )
    for model in all_models:
        model_data = {
            "id": model,
            "object": "model",
            "created": 1678888888,
            "owned_by": "openai",
            "permission": [],
            "root": model,
            "parent": None
        }
        detailed_models.append(model_data)
        if "DeepSeek-R1" in model:
            detailed_models.append({
                "id": model + "-thinking",
                "object": "model",
                "created": 1678888888,
                "owned_by": "openai",
                "permission": [],
                "root": model + "-thinking",
                "parent": None
            })
            detailed_models.append({
                "id": model + "-openwebui",
                "object": "model",
                "created": 1678888888,
                "owned_by": "openai",
                "permission": [],
                "root": model + "-openwebui",
                "parent": None
            })
    return jsonify({
        "success": True,
        "data": detailed_models
    })

# 获取账单使用情况的路由，去除了 /handsome 前缀
@app.route('/v1/dashboard/billing/usage', methods=['GET'])
def billing_usage():
    if not check_authorization(request):
        return jsonify({"error": "Unauthorized"}), 401
    daily_usage = []
    return jsonify({
        "object": "list",
        "data": daily_usage,
        "total_usage": 0
    })

# 获取账单订阅信息的路由，去除了 /handsome 前缀
@app.route('/v1/dashboard/billing/subscription', methods=['GET'])
def billing_subscription():
    if not check_authorization(request):
        return jsonify({"error": "Unauthorized"}), 401
    keys = valid_keys_global + unverified_keys_global
    total_balance = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10000) as executor:
        futures = [
            executor.submit(get_credit_summary, key) for key in keys
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                credit_summary = future.result()
                if credit_summary:
                    total_balance += credit_summary.get("total_balance", 0)
            except Exception as exc:
                logging.error(f"获取额度信息生成异常: {exc}")
    return jsonify({
        "object": "billing_subscription",
        "access_until": int(datetime(9999, 12, 31).timestamp()),
        "soft_limit": 0,
        "hard_limit": total_balance,
        "system_hard_limit": total_balance,
        "soft_limit_usd": 0,
        "hard_limit_usd": total_balance,
        "system_hard_limit_usd": total_balance
    })

# 处理嵌入请求的路由，去除了 /handsome 前缀
@app.route('/v1/embeddings', methods=['POST'])
def handsome_embeddings():
    if not check_authorization(request):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    if not data or 'model' not in data:
        return jsonify({"error": "Invalid request data"}), 400
    if data['model'] not in models["embedding"]:
        return jsonify({"error": "Invalid model"}), 400
    model_name = data['model']
    request_type = determine_request_type(
        model_name,
        models["embedding"],
        models["free_embedding"]
    )
    api_key = select_key(request_type, model_name)
    if not api_key:
        return jsonify(
            {"error": ("No available API key for this request type or all keys have reached their limits")}), 429
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        start_time = time.time()
        response = requests.post(
            EMBEDDINGS_ENDPOINT,
            headers=headers,
            json=data,
            timeout=120
        )
        if response.status_code == 429:
            return jsonify(response.json()), 429
        response.raise_for_status()
        end_time = time.time()
        response_json = response.json()
        total_time = end_time - start_time
        try:
            prompt_tokens = response_json["usage"]["prompt_tokens"]
            embedding_data = response_json["data"]
        except (KeyError, ValueError, IndexError) as e:
            logging.error(
                f"解析响应 JSON 失败: {e}, "
                f"完整内容: {response_json}"
            )
            prompt_tokens = 0
            embedding_data = []
        logging.info(
            f"使用的key: {api_key}, "
            f"提示token: {prompt_tokens}, "
            f"总共用时: {total_time:.4f}秒, "
            f"使用的模型: {model_name}"
        )
        with data_lock:
            request_timestamps.append(time.time())
            token_counts.append(prompt_tokens)
            request_timestamps_day.append(time.time())
            token_counts_day.append(prompt_tokens)
        return jsonify({
            "object": "list",
            "data": embedding_data,
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens
            }
        })
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

# 处理图像生成请求的路由，去除了 /handsome 前缀
@app.route('/v1/images/generations', methods=['POST'])
def handsome_images_generations():
    if not check_authorization(request):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    if not data or 'model' not in data:
        return jsonify({"error": "Invalid request data"}), 400
    if data['model'] not in models["image"]:
        return jsonify({"error": "Invalid model"}), 400
    model_name = data.get('model')
    request_type = determine_request_type(
        model_name,
        models["image"],
        models["free_image"]
    )
    api_key = select_key(request_type, model_name)
    if not api_key:
        return jsonify(
            {"error": ("No available API key for this request type or all keys have reached their limits")}), 429
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response_data = {}
    if "stable-diffusion" in model_name or model_name in ["black-forest-labs/FLUX.1-schnell",
                                                          "Pro/black-forest-labs/FLUX.1-schnell",
                                                          "black-forest-labs/FLUX.1-dev",
                                                          "black-forest-labs/FLUX.1-pro"]:
        siliconflow_data = get_siliconflow_data(model_name, data)
        try:
            start_time = time.time()
            response = requests.post(
                IMAGE_ENDPOINT,
                headers=headers,
                json=siliconflow_data,
                timeout=120
            )
            if response.status_code == 429:
                return jsonify(response.json()), 429
            response.raise_for_status()
            end_time = time.time()
            response_json = response.json()
            total_time = end_time - start_time
            try:
                images = response_json.get("images", [])
                openai_images = []
                for item in images:
                    if isinstance(item, dict) and "url" in item:
                        image_url = item["url"]
                        print(f"image_url: {image_url}")
                        if data.get("response_format") == "b64_json":
                            try:
                                image_data = session.get(image_url, stream=True).raw
                                image = Image.open(image_data)
                                buffered = io.BytesIO()
                                image.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                openai_images.append({"b64_json": img_str})
                            except Exception as e:
                                logging.error(f"图片转base64失败: {e}")
                                openai_images.append({"url": image_url})
                        else:
                            openai_images.append({"url": image_url})
                    else:
                        logging.error(f"无效的图片数据: {item}")
                        openai_images.append({"url": item})
                response_data = {
                    "created": int(time.time()),
                    "data": openai_images
                }
            except (KeyError, ValueError, IndexError) as e:
                logging.error(
                    f"解析响应 JSON 失败: {e}, "
                    f"完整内容: {response_json}"
                )
                response_data = {
                    "created": int(time.time()),
                    "data": []
                }
            logging.info(
                f"使用的key: {api_key}, "
                f"总共用时: {total_time:.4f}秒, "
                f"使用的模型: {model_name}"
            )
            with data_lock:
                request_timestamps.append(time.time())
                token_counts.append(0)
                request_timestamps_day.append(time.time())
                token_counts_day.append(0)
            return jsonify(response_data)
        except requests.exceptions.RequestException as e:
            logging.error(f"请求转发异常: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unsupported model"}), 400

# 处理聊天完成请求的路由，去除了 /handsome 前缀
@app.route('/v1/chat/completions', methods=['POST'])
def handsome_chat_completions():
    if not check_authorization(request):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    if not data or 'model' not in data:
        return jsonify({"error": "Invalid request data"}), 400
    model_name = data['model']
    if model_name not in models["text"] and model_name not in models["image"]:
        if "DeepSeek-R1" in model_name and (model_name.endswith("-openwebui") or model_name.endswith("-thinking")):
            pass
        else:
            return jsonify({"error": "Invalid model"}), 400
    model_realname = model_name.replace("-thinking", "").replace("-openwebui", "")
    request_type = determine_request_type(
        model_realname,
        models["text"] + models["image"],
        models["free_text"] + models["free_image"]
    )
    api_key = select_key(request_type, model_name)
    if not api_key:
        return jsonify(
            {
                "error": (
                    "No available API key for this "
                    "request type or all keys have "
                    "reached their limits"
                )
            }
        ), 429
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if "DeepSeek-R1" in model_name and ("thinking" in model_name or "openwebui" in model_name):
        data['model'] = model_realname
        start_time = time.time()
        response = requests.post(
            TEST_MODEL_ENDPOINT,
            headers=headers,
            json=data,
            stream=data.get("stream", False),
            timeout=120
        )
        if response.status_code == 429:
            return jsonify(response.json()), 429
        if data.get("stream", False):
            def generate():
                if model_name.endswith("-openwebui"):
                    first_chunk_time = None
                    full_response_content = ""
                    reasoning_content_accumulated = ""
                    content_accumulated = ""
                    first_reasoning_chunk = True
                    for chunk in response.iter_lines():
                        if chunk:
                            if first_chunk_time is None:
                                first_chunk_time = time.time()
                            full_response_content += chunk.decode("utf-8")
                            for line in chunk.decode("utf-8").splitlines():
                                if line.startswith("data:"):
                                    try:
                                        chunk_json = json.loads(line.lstrip("data: ").strip())
                                        if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                            delta = chunk_json["choices"][0].get("delta", {})
                                            if delta.get("reasoning_content") is not None:
                                                reasoning_chunk = delta["reasoning_content"]
                                                if first_reasoning_chunk:
                                                    think_chunk = f"<"
                                                    yield f"data: {json.dumps({'choices': [{'delta': {'content': think_chunk}, 'index': 0}]})}\n\n"
                                                    think_chunk = f"think"
                                                    yield f"data: {json.dumps({'choices': [{'delta': {'content': think_chunk}, 'index': 0}]})}\n\n"
                                                    think_chunk = f">\n"
                                                    yield f"data: {json.dumps({'choices': [{'delta': {'content': think_chunk}, 'index': 0}]})}\n\n"
                                                    first_reasoning_chunk = False
                                                yield f"data: {json.dumps({'choices': [{'delta': {'content': reasoning_chunk}, 'index': 0}]})}\n\n"
                                            if delta.get("content") is not None:
                                                if not first_reasoning_chunk:
                                                    reasoning_chunk = f"\n</think>\n"
                                                    yield f"data: {json.dumps({'choices': [{'delta': {'content': reasoning_chunk}, 'index': 0}]})}\n\n"
                                                    first_reasoning_chunk = True
                                                yield f"data: {json.dumps({'choices': [{'delta': {'content': delta["content"]}, 'index': 0}]})}\n\n"
                                    except (KeyError, ValueError, json.JSONDecodeError) as e:
                                        continue
                    end_time = time.time()
                    first_token_time = (
                        first_chunk_time - start_time
                        if first_chunk_time else 0
                    )
                    total_time = end_time - start_time
                    prompt_tokens = 0
                    completion_tokens = 0
                    for line in full_response_content.splitlines():
                        if line.startswith("data:"):
                            line = line[5:].strip()
                            if line == "[DONE]":
                                continue
                            try:
                                response_json = json.loads(line)
                                if (
                                        "usage" in response_json and
                                        "completion_tokens" in response_json["usage"]
                                ):
                                    completion_tokens += response_json[
                                        "usage"
                                    ]["completion_tokens"]
                                if (
                                        "usage" in response_json and
                                        "prompt_tokens" in response_json["usage"]
                                ):
                                    prompt_tokens = response_json[
                                        "usage"
                                    ]["prompt_tokens"]
                            except (KeyError, ValueError, IndexError) as e:
                                pass
                    user_content = ""
                    messages = data.get("messages", [])
                    for message in messages:
                        if message["role"] == "user":
                            if isinstance(message["content"], str):
                                user_content += message["content"] + " "
                            elif isinstance(message["content"], list):
                                for item in message["content"]:
                                    if (
                                            isinstance(item, dict) and
                                            item.get("type") == "text"
                                    ):
                                        user_content += (
                                                item.get("text", "") +
                                                " "
                                        )
                    user_content = user_content.strip()
                    user_content_replaced = user_content.replace(
                        '\n', '\\n'
                    ).replace('\r', '\\n')
                    response_content_replaced = (
                                                            f"```Thinking\n{reasoning_content_accumulated}\n```\n" if reasoning_content_accumulated else "") + content_accumulated
                    response_content_replaced = response_content_replaced.replace(
                        '\n', '\\n'
                    ).replace('\r', '\\n')
                    logging.info(
                        f"使用的key: {api_key}, "
                        f"提示token: {prompt_tokens}, "
                        f"输出token: {completion_tokens}, "
                        f"首字用时: {first_token_time:.4f}秒, "
                        f"总共用时: {total_time:.4f}秒, "
                        f"使用的模型: {model_name}, "
                        f"用户的内容: {user_content_replaced}, "
                        f"输出的内容: {response_content_replaced}"
                    )
                    with data_lock:
                        request_timestamps.append(time.time())
                        token_counts.append(prompt_tokens + completion_tokens)
                    yield "data: [DONE]\n\n"
                    return Response(
                        stream_with_context(generate()),
                        content_type="text/event-stream"
                    )
                first_chunk_time = None
                full_response_content = ""
                reasoning_content_accumulated = ""
                content_accumulated = ""
                first_reasoning_chunk = True
                for chunk in response.iter_lines():
                    if chunk:
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                        full_response_content += chunk.decode("utf-8")
                        for line in chunk.decode("utf-8").splitlines():
                            if line.startswith("data:"):
                                try:
                                    chunk_json = json.loads(line.lstrip("data: ").strip())
                                    if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                        delta = chunk_json["choices"][0].get("delta", {})
                                        if delta.get("reasoning_content") is not None:
                                            reasoning_chunk = delta["reasoning_content"]
                                            reasoning_chunk = reasoning_chunk.replace('\n', '\n> ')
                                            if first_reasoning_chunk:
                                                reasoning_chunk = "> " + reasoning_chunk
                                                first_reasoning_chunk = False
                                            yield f"data: {json.dumps({'choices': [{'delta': {'content': reasoning_chunk}, 'index': 0}]})}\n\n"
                                        if delta.get("content") is not None:
                                            if not first_reasoning_chunk:
                                                yield f"data: {json.dumps({'choices': [{'delta': {'content': '\n\n'}, 'index': 0}]})}\n\n"
                                                first_reasoning_chunk = True
                                            yield f"data: {json.dumps({'choices': [{'delta': {'content': delta["content"]}, 'index': 0}]})}\n\n"
                                except (KeyError, ValueError, json.JSONDecodeError) as e:
                                    continue
                end_time = time.time()
                first_token_time = (
                        first_chunk_time - start_time
                        if first_chunk_time else 0
                )
                total_time = end_time - start_time
                prompt_tokens = 0
                completion_tokens = 0
                for line in full_response_content.splitlines():
                    if line.startswith("data:"):
                        line = line[5:].strip()
                        if line == "[DONE]":
                            continue
                        try:
                            response_json = json.loads(line)
                            if (
                                    "usage" in response_json and
                                    "completion_tokens" in response_json["usage"]
                            ):
                                completion_tokens += response_json[
                                    "usage"
                                ]["completion_tokens"]
                            if (
                                    "usage" in response_json and
                                    "prompt_tokens" in response_json["usage"]
                            ):
                                prompt_tokens = response_json[
                                    "usage"
                                ]["prompt_tokens"]
                        except (KeyError, ValueError, IndexError) as e:
                            pass
                user_content = ""
                messages = data.get("messages", [])
                for message in messages:
                    if message["role"] == "user":
                        if isinstance(message["content"], str):
                            user_content += message["content"] + " "
                        elif isinstance(message["content"], list):
                            for item in message["content"]:
                                if (
                                        isinstance(item, dict) and
                                        item.get("type") == "text"
                                ):
                                    user_content += (
                                            item.get("text", "") +
                                            " "
                                    )
                user_content = user_content.strip()
                user_content_replaced = user_content.replace(
                    '\n', '\\n'
                ).replace('\r', '\\n')
                response_content_replaced = (
                                                    f"```Thinking\n{reasoning_content_accumulated}\n```\n" if reasoning_content_accumulated else "") + content_accumulated
                response_content_replaced = response_content_replaced.replace(
                    '\n', '\\n'
                ).replace('\r', '\\n')
                logging.info(
                    f"使用的key: {api_key}, "
                    f"提示token: {prompt_tokens}, "
                    f"输出token: {completion_tokens}, "
                    f"首字用时: {first_token_time:.4f}秒, "
                    f"总共用时: {total_time:.4f}秒, "
                    f"使用的模型: {model_name}, "
                    f"用户的内容: {user_content_replaced}, "
                    f"输出的内容: {response_content_replaced}"
                )
                with data_lock:
                    request_timestamps.append(time.time())
                    token_counts.append(prompt_tokens + completion_tokens)
                yield "data: [DONE]\n\n"
            return Response(
                stream_with_context(generate()),
                content_type="text/event-stream"
            )
        else:
            response.raise_for_status()
            end_time = time.time()
            response_json = response.json()
            total_time = end_time - start_time
            try:
                prompt_tokens = response_json["usage"]["prompt_tokens"]
                completion_tokens = response_json["usage"]["completion_tokens"]
                response_content = ""
                if model_name.endswith("-thinking") and "choices" in response_json and len(
                        response_json["choices"]) > 0:
                    choice = response_json["choices"][0]
                    if "message" in choice:
                        if "reasoning_content" in choice["message"]:
                            reasoning_content = choice["message"]["reasoning_content"]
                            reasoning_content = reasoning_content.replace('\n', '\n> ')
                            reasoning_content = '> ' + reasoning_content
                            formatted_reasoning = f"{reasoning_content}\n"
                            response_content += formatted_reasoning + "\n"
                        if "content" in choice["message"]:
                            response_content += choice["message"]["content"]
                elif model_name.endswith("-openwebui") and "choices" in response_json and len(
                        response_json["choices"]) > 0:
                    choice = response_json["choices"][0]
                    if "message" in choice:
                        if "reasoning_content" in choice["message"]:
                            reasoning_content = choice["message"]["reasoning_content"]
                            response_content += f"<think>\n{reasoning_content}\n</think>\n"
                        if "content" in choice["message"]:
                            response_content += choice["message"]["content"]
            except (KeyError, ValueError, IndexError) as e:
                logging.error(
                    f"解析非流式响应 JSON 失败: {e}, "
                    f"完整内容: {response_json}"
                )
                prompt_tokens = 0
                completion_tokens = 0
                response_content = ""
            user_content = ""
            messages = data.get("messages", [])
            for message in messages:
                if message["role"] == "user":
                    if isinstance(message["content"], str):
                        user_content += message["content"] + " "
                    elif isinstance(message["content"], list):
                        for item in message["content"]:
                            if (
                                    isinstance(item, dict) and
                                    item.get("type") == "text"
                            ):
                                user_content += (
                                        item.get("text", "") +
                                        " "
                                )
            user_content = user_content.strip()
            user_content_replaced = user_content.replace(
                '\n', '\\n'
            ).replace('\r', '\\n')
            response_content_replaced = response_content.replace(
                '\n', '\\n'
            ).replace('\r', '\\n')
            logging.info(
                f"使用的key: {api_key}, "
                f"提示token: {prompt_tokens}, "
                f"输出token: {completion_tokens}, "
                f"首字用时: 0, "
                f"总共用时: {total_time:.4f}秒, "
                f"使用的模型: {model_name}, "
                f"用户的内容: {user_content_replaced}, "
                f"输出的内容: {response_content_replaced}"
            )
            with data_lock:
                request_timestamps.append(time.time())
                token_counts.append(prompt_tokens + completion_tokens)
            formatted_response = {
                "id": response_json.get("id", ""),
                "object": "chat.completion",
                "created": response_json.get("created", int(time.time())),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            return jsonify(formatted_response)
    if model_name in models["image"]:
        if isinstance(data.get("messages"), list):
            data = data.copy()
            data["prompt"] = extract_user_content(data["messages"])
        siliconflow_data = get_siliconflow_data(model_name, data)
        try:
            start_time = time.time()
            response = requests.post(
                IMAGE_ENDPOINT,
                headers=headers,
                json=siliconflow_data,
                stream=data.get("stream", False)
            )
            if response.status_code == 429:
                return jsonify(response.json()), 429
            if data.get("stream", False):
                def generate():
                    try:
                        response.raise_for_status()
                        response_json = response.json()
                        images = response_json.get("images", [])
                        image_url = ""
                        if images and isinstance(images[0], dict) and "url" in images[0]:
                            image_url = images[0]["url"]
                            logging.info(f"Extracted image URL: {image_url}")
                        elif images and isinstance(images[0], str):
                            image_url = images[0]
                            logging.info(f"Extracted image URL: {image_url}")
                        markdown_image_link = create_base64_markdown_image(image_url)
                        if image_url:
                            chunk_size = 8192
                            for i in range(0, len(markdown_image_link), chunk_size):
                                chunk = markdown_image_link[i:i + chunk_size]
                                chunk_data = {
                                    "id": f"chatcmpl-{uuid.uuid4()}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "role": "assistant",
                                                "content": chunk
                                            },
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8')
                        else:
                            chunk_data = {
                                "id": f"chatcmpl-{uuid.uuid4()}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": "Failed to generate image"
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8')
                        end_chunk_data = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(end_chunk_data)}\n\n".encode('utf-8')
                        with data_lock:
                            request_timestamps.append(time.time())
                            token_counts.append(0)
                            request_timestamps_day.append(time.time())
                            token_counts_day.append(0)
                    except requests.exceptions.RequestException as e:
                        logging.error(f"请求转发异常: {e}")
                        error_chunk_data = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": f"Error: {str(e)}"
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(error_chunk_data)}\n\n".encode('utf-8')
                        end_chunk_data = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(end_chunk_data)}\n\n".encode('utf-8')
                    logging.info(
                        f"使用的key: {api_key}, "
                        f"使用的模型: {model_name}"
                    )
                    yield "data: [DONE]\n\n".encode('utf-8')
                return Response(stream_with_context(generate()), content_type='text/event-stream')
            else:
                response.raise_for_status()
                end_time = time.time()
                response_json = response.json()
                total_time = end_time - start_time
                try:
                    images = response_json.get("images", [])
                    image_url = ""
                    if images and isinstance(images[0], dict) and "url" in images[0]:
                        image_url = images[0]["url"]
                        logging.info(f"Extracted image URL: {image_url}")
                    elif images and isinstance(images[0], str):
                        image_url = images[0]
                        logging.info(f"Extracted image URL: {image_url}")
                    markdown_image_link = f"![image]({image_url})"
                    response_data = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": markdown_image_link if image_url else "Failed to generate image",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                except (KeyError, ValueError, IndexError) as e:
                    logging.error(
                        f"解析响应 JSON 失败: {e}, "
                        f"完整内容: {response_json}"
                    )
                    response_data = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Failed to process image data",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                logging.info(
                    f"使用的key: {api_key}, "
                    f"总共用时: {total_time:.4f}秒, "
                    f"使用的模型: {model_name}"
                )
                with data_lock:
                    request_timestamps.append(time.time())
                    token_counts.append(0)
                    request_timestamps_day.append(time.time())
                    token_counts_day.append(0)
                return jsonify(response_data)
        except requests.exceptions.RequestException as e:
            logging.error(f"请求转发异常: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        try:
            start_time = time.time()
            response = requests.post(
                TEST_MODEL_ENDPOINT,
                headers=headers,
                json=data,
                stream=data.get("stream", False)
            )
            if response.status_code == 429:
                return jsonify(response.json()), 429
            if data.get("stream", False):
                def generate():
                    first_chunk_time = None
                    full_response_content = ""
                    for chunk in response.iter_content(chunk_size=2048):
                        if chunk:
                            if first_chunk_time is None:
                                first_chunk_time = time.time()
                            full_response_content += chunk.decode("utf-8")
                            yield chunk
                    end_time = time.time()
                    first_token_time = (
                            first_chunk_time - start_time
                            if first_chunk_time else 0
                    )
                    total_time = end_time - start_time
                    prompt_tokens = 0
                    completion_tokens = 0
                    response_content = ""
                    for line in full_response_content.splitlines():
                        if line.startswith("data:"):
                            line = line[5:].strip()
                            if line == "[DONE]":
                                continue
                            try:
                                response_json = json.loads(line)
                                if (
                                        "usage" in response_json and
                                        "completion_tokens" in response_json["usage"]
                                ):
                                    completion_tokens = response_json[
                                        "usage"
                                    ]["completion_tokens"]
                                if (
                                        "choices" in response_json and
                                        len(response_json["choices"]) > 0 and
                                        "delta" in response_json["choices"][0] and
                                        "content" in response_json[
                                            "choices"
                                        ][0]["delta"]
                                ):
                                    response_content += response_json[
                                        "choices"
                                    ][0]["delta"]["content"]
                                if (
                                        "usage" in response_json and
                                        "prompt_tokens" in response_json["usage"]
                                ):
                                    prompt_tokens = response_json[
                                        "usage"
                                    ]["prompt_tokens"]
                            except (
                                    KeyError,
                                    ValueError,
                                    IndexError
                            ) as e:
                                logging.error(
                                    f"解析流式响应单行 JSON 失败: {e}, "
                                    f"行内容: {line}"
                                )
                    user_content = extract_user_content(data.get("messages", []))
                    user_content_replaced = user_content.replace(
                        '\n', '\\n'
                    ).replace('\r', '\\n')
                    response_content_replaced = response_content.replace(
                        '\n', '\\n'
                    ).replace('\r', '\\n')
                    logging.info(
                        f"使用的key: {api_key}, "
                        f"提示token: {prompt_tokens}, "
                        f"输出token: {completion_tokens}, "
                        f"首字用时: {first_token_time:.4f}秒, "
                        f"总共用时: {total_time:.4f}秒, "
                        f"使用的模型: {model_name}, "
                        f"用户的内容: {user_content_replaced}, "
                        f"输出的内容: {response_content_replaced}"
                    )
                    with data_lock:
                        request_timestamps.append(time.time())
                        token_counts.append(prompt_tokens + completion_tokens)
                        request_timestamps_day.append(time.time())
                        token_counts_day.append(prompt_tokens + completion_tokens)
                return Response(
                    stream_with_context(generate()),
                    content_type=response.headers['Content-Type']
                )
            else:
                response.raise_for_status()
                end_time = time.time()
                response_json = response.json()
                total_time = end_time - start_time
                try:
                    prompt_tokens = response_json["usage"]["prompt_tokens"]
                    completion_tokens = response_json[
                        "usage"
                    ]["completion_tokens"]
                    response_content = response_json[
                        "choices"
                    ][0]["message"]["content"]
                except (KeyError, ValueError, IndexError) as e:
                    logging.error(
                        f"解析非流式响应 JSON 失败: {e}, "
                        f"完整内容: {response_json}"
                    )
                    prompt_tokens = 0
                    completion_tokens = 0
                    response_content = ""
                user_content = extract_user_content(data.get("messages", []))
                user_content_replaced = user_content.replace(
                    '\n', '\\n'
                ).replace('\r', '\\n')
                response_content_replaced = response_content.replace(
                    '\n', '\\n'
                ).replace('\r', '\\n')
                logging.info(
                    f"使用的key: {api_key}, "
                    f"提示token: {prompt_tokens}, "
                    f"输出token: {completion_tokens}, "
                    f"首字用时: 0, "
                    f"总共用时: {total_time:.4f}秒, "
                    f"使用的模型: {model_name}, "
                    f"用户的内容: {user_content_replaced}, "
                    f"输出的内容: {response_content_replaced}"
                )
                with data_lock:
                    request_timestamps.append(time.time())
                    if "prompt_tokens" in response_json["usage"] and "completion_tokens" in response_json["usage"]:
                        token_counts.append(
                            response_json["usage"]["prompt_tokens"] + response_json["usage"]["completion_tokens"])
                    else:
                        token_counts.append(0)
                    request_timestamps_day.append(time.time())
                    if "prompt_tokens" in response_json["usage"] and "completion_tokens" in response_json["usage"]:
                        token_counts_day.append(
                            response_json["usage"]["prompt_tokens"] + response_json["usage"]["completion_tokens"])
                    else:
                        token_counts_day.append(0)
                return jsonify(response_json)
        except requests.exceptions.RequestException as e:
            logging.error(f"请求转发异常: {e}")
            return jsonify({"error": str(e)}), 500

# 初始化代码
load_keys()
logging.info("程序启动时首次加载 keys 已执行")
scheduler.start()
logging.info("首次加载 keys 已手动触发执行")
refresh_models()
logging.info("首次刷新模型列表已手动触发执行")

if __name__ == '__main__':
    # 注意：在 Vercel 上部署时，不需要 app.run()。
    # 这里保留 app.run 是为了方便本地调试
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
