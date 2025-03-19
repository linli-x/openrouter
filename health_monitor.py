import os
import time
import psutil
import logging
import threading
import json
from datetime import datetime, timedelta

class HealthMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.system_metrics = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": []
        }
        self.api_metrics = {
            "key_status": {},
            "request_count": [],
            "error_count": [],
            "response_time": []
        }
        self.max_history_points = 1440  # 存储24小时的数据，每分钟一个点
        self.alert_thresholds = {
            "cpu": 90,  # CPU使用率超过90%报警
            "memory": 90,  # 内存使用率超过90%报警
            "disk": 90,  # 磁盘使用率超过90%报警
            "response_time": 5,  # 响应时间超过5秒报警
            "error_rate": 10  # 错误率超过10%报警
        }
        self.alerts = []
        self.max_alerts = 100
        
    def collect_system_metrics(self):
        """收集系统指标"""
        try:
            # 收集CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 收集内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 收集磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 收集网络IO统计
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent
            net_recv = net_io.bytes_recv
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with self.lock:
                # 添加新的指标数据
                self.system_metrics["cpu"].append({"timestamp": timestamp, "value": cpu_percent})
                self.system_metrics["memory"].append({"timestamp": timestamp, "value": memory_percent})
                self.system_metrics["disk"].append({"timestamp": timestamp, "value": disk_percent})
                self.system_metrics["network"].append({"timestamp": timestamp, "sent": net_sent, "recv": net_recv})
                
                # 限制历史数据点数量
                for key in ["cpu", "memory", "disk", "network"]:
                    if len(self.system_metrics[key]) > self.max_history_points:
                        self.system_metrics[key] = self.system_metrics[key][-self.max_history_points:]
                
                # 检查是否需要触发告警
                self._check_system_alerts(cpu_percent, memory_percent, disk_percent)
                
            return {
                "cpu": cpu_percent,
                "memory": memory_percent,
                "disk": disk_percent,
                "network": {"sent": net_sent, "recv": net_recv}
            }
        except Exception as e:
            logging.error(f"收集系统指标时发生错误: {str(e)}")
            return None
    
    def update_api_metrics(self, key_status, request_count=None, error_count=None, response_time=None):
        """更新API指标"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with self.lock:
                # 更新密钥状态
                self.api_metrics["key_status"] = key_status
                
                # 更新请求计数
                if request_count is not None:
                    self.api_metrics["request_count"].append({"timestamp": timestamp, "value": request_count})
                    if len(self.api_metrics["request_count"]) > self.max_history_points:
                        self.api_metrics["request_count"] = self.api_metrics["request_count"][-self.max_history_points:]
                
                # 更新错误计数
                if error_count is not None:
                    self.api_metrics["error_count"].append({"timestamp": timestamp, "value": error_count})
                    if len(self.api_metrics["error_count"]) > self.max_history_points:
                        self.api_metrics["error_count"] = self.api_metrics["error_count"][-self.max_history_points:]
                
                # 更新响应时间
                if response_time is not None:
                    self.api_metrics["response_time"].append({"timestamp": timestamp, "value": response_time})
                    if len(self.api_metrics["response_time"]) > self.max_history_points:
                        self.api_metrics["response_time"] = self.api_metrics["response_time"][-self.max_history_points:]
                
                # 检查API指标告警
                if response_time is not None and request_count is not None and error_count is not None:
                    self._check_api_alerts(response_time, request_count, error_count)
        except Exception as e:
            logging.error(f"更新API指标时发生错误: {str(e)}")
    
    def _check_system_alerts(self, cpu_percent, memory_percent, disk_percent):
        """检查系统指标是否需要触发告警"""
        current_time = datetime.now()
        
        if cpu_percent > self.alert_thresholds["cpu"]:
            self._add_alert("CPU使用率过高", f"当前CPU使用率: {cpu_percent}%，超过阈值{self.alert_thresholds['cpu']}%", "high")
        
        if memory_percent > self.alert_thresholds["memory"]:
            self._add_alert("内存使用率过高", f"当前内存使用率: {memory_percent}%，超过阈值{self.alert_thresholds['memory']}%", "high")
        
        if disk_percent > self.alert_thresholds["disk"]:
            self._add_alert("磁盘使用率过高", f"当前磁盘使用率: {disk_percent}%，超过阈值{self.alert_thresholds['disk']}%", "high")
    
    def _check_api_alerts(self, response_time, request_count, error_count):
        """检查API指标是否需要触发告警"""
        if response_time > self.alert_thresholds["response_time"]:
            self._add_alert("API响应时间过长", f"当前响应时间: {response_time}秒，超过阈值{self.alert_thresholds['response_time']}秒", "medium")
        
        if request_count > 0:
            error_rate = (error_count / request_count) * 100
            if error_rate > self.alert_thresholds["error_rate"]:
                self._add_alert("API错误率过高", f"当前错误率: {error_rate:.2f}%，超过阈值{self.alert_thresholds['error_rate']}%", "high")
    
    def _add_alert(self, title, message, severity):
        """添加一个新的告警"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = {
            "timestamp": current_time,
            "title": title,
            "message": message,
            "severity": severity
        }
        
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        logging.warning(f"健康监测告警: {title} - {message}")
    
    def get_health_report(self):
        """获取健康报告"""
        with self.lock:
            # 获取最新的系统指标
            latest_cpu = self.system_metrics["cpu"][-1]["value"] if self.system_metrics["cpu"] else 0
            latest_memory = self.system_metrics["memory"][-1]["value"] if self.system_metrics["memory"] else 0
            latest_disk = self.system_metrics["disk"][-1]["value"] if self.system_metrics["disk"] else 0
            
            # 计算平均响应时间（最近10分钟）
            recent_response_times = [item["value"] for item in self.api_metrics["response_time"][-10:]] if self.api_metrics["response_time"] else []
            avg_response_time = sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0
            
            # 计算错误率（最近10分钟）
            recent_requests = sum([item["value"] for item in self.api_metrics["request_count"][-10:]]) if self.api_metrics["request_count"] else 0
            recent_errors = sum([item["value"] for item in self.api_metrics["error_count"][-10:]]) if self.api_metrics["error_count"] else 0
            error_rate = (recent_errors / recent_requests * 100) if recent_requests > 0 else 0
            
            # 获取密钥状态统计
            key_stats = {
                "total": sum(len(keys) for keys in self.api_metrics["key_status"].values()),
                "valid": len(self.api_metrics["key_status"].get("valid", [])),
                "free": len(self.api_metrics["key_status"].get("free", [])),
                "invalid": len(self.api_metrics["key_status"].get("invalid", [])),
                "unverified": len(self.api_metrics["key_status"].get("unverified", []))
            }
            
            # 计算系统健康状态
            system_status = "healthy"
            if latest_cpu > self.alert_thresholds["cpu"] or \
               latest_memory > self.alert_thresholds["memory"] or \
               latest_disk > self.alert_thresholds["disk"]:
                system_status = "warning"
            
            # 计算API健康状态
            api_status = "healthy"
            if avg_response_time > self.alert_thresholds["response_time"] or \
               error_rate > self.alert_thresholds["error_rate"]:
                api_status = "warning"
            
            # 如果没有有效的密钥，API状态为危险
            if key_stats["valid"] == 0:
                api_status = "danger"
            
            # 整体健康状态
            overall_status = "healthy"
            if system_status == "warning" or api_status == "warning":
                overall_status = "warning"
            if api_status == "danger":
                overall_status = "danger"
            
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_status": overall_status,
                "system": {
                    "status": system_status,
                    "cpu": latest_cpu,
                    "memory": latest_memory,
                    "disk": latest_disk
                },
                "api": {
                    "status": api_status,
                    "avg_response_time": avg_response_time,
                    "error_rate": error_rate,
                    "key_stats": key_stats
                },
                "alerts": self.alerts[-5:]  # 最近5条告警
            }
    
    def get_detailed_metrics(self, metric_type, time_range="1h"):
        """获取详细的指标数据用于图表显示"""
        with self.lock:
            now = datetime.now()
            
            # 根据时间范围确定起始时间
            if time_range == "1h":
                start_time = now - timedelta(hours=1)
            elif time_range == "6h":
                start_time = now - timedelta(hours=6)
            elif time_range == "24h":
                start_time = now - timedelta(hours=24)
            else:  # 默认1小时
                start_time = now - timedelta(hours=1)
            
            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 根据指标类型获取相应数据
            if metric_type in ["cpu", "memory", "disk"]:
                # 系统指标
                data = [item for item in self.system_metrics[metric_type] 
                        if item["timestamp"] >= start_time_str]
                return {
                    "type": metric_type,
                    "unit": "count",
                    "data": data
                }
            elif metric_type == "response_time":
                # API响应时间
                data = [item for item in self.api_metrics["response_time"] 
                        if item["timestamp"] >= start_time_str]
                return {
                    "type": metric_type,
                    "unit": "seconds",
                    "data": data
                }
            else:
                return {"error": "Invalid metric type"}

# 创建健康监测实例
health_monitor = HealthMonitor()

# 后台线程定期收集系统指标
def background_metrics_collector():
    while True:
        try:
            health_monitor.collect_system_metrics()
            time.sleep(60)  # 每分钟收集一次
        except Exception as e:
            logging.error(f"后台指标收集线程发生错误: {str(e)}")
            time.sleep(60)  # 发生错误后等待一分钟再继续

# 启动后台收集线程
metrics_thread = threading.Thread(target=background_metrics_collector, daemon=True)
metrics_thread.start(),
                    "unit": "%",
                    "data": data
                }
            elif metric_type == "network":
                # 网络指标
                data = [item for item in self.system_metrics["network"] 
                        if item["timestamp"] >= start_time_str]
                return {
                    "type": metric_type,
                    "unit": "count",
                    "data": data
                }
            elif metric_type == "response_time":
                # API响应时间
                data = [item for item in self.api_metrics["response_time"] 
                        if item["timestamp"] >= start_time_str]
                return {
                    "type": metric_type,
                    "unit": "seconds",
                    "data": data
                }
            else:
                return {"error": "Invalid metric type"}

# 创建健康监测实例
health_monitor = HealthMonitor()

# 后台线程定期收集系统指标
def background_metrics_collector():
    while True:
        try:
            health_monitor.collect_system_metrics()
            time.sleep(60)  # 每分钟收集一次
        except Exception as e:
            logging.error(f"后台指标收集线程发生错误: {str(e)}")
            time.sleep(60)  # 发生错误后等待一分钟再继续

# 启动后台收集线程
metrics_thread = threading.Thread(target=background_metrics_collector, daemon=True)
metrics_thread.start(),
                    "unit": "bytes",
                    "data": data
                }
            elif metric_type in ["request_count", "error_count"]:
                # API请求和错误计数
                data = [item for item in self.api_metrics[metric_type] 
                        if item["timestamp"] >= start_time_str]
                return {
                    "type": metric_type,
                    "unit": "count",
                    "data": data
                }
            elif metric_type == "response_time":
                # API响应时间
                data = [item for item in self.api_metrics["response_time"] 
                        if item["timestamp"] >= start_time_str]
                return {
                    "type": metric_type,
                    "unit": "seconds",
                    "data": data
                }
            else:
                return {"error": "Invalid metric type"}

# 创建健康监测实例
health_monitor = HealthMonitor()

# 后台线程定期收集系统指标
def background_metrics_collector():
    while True:
        try:
            health_monitor.collect_system_metrics()
            time.sleep(60)  # 每分钟收集一次
        except Exception as e:
            logging.error(f"后台指标收集线程发生错误: {str(e)}")
            time.sleep(60)  # 发生错误后等待一分钟再继续

# 启动后台收集线程
metrics_thread = threading.Thread(target=background_metrics_collector, daemon=True)
metrics_thread.start()