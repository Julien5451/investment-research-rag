"""HTTP-only client for the demo; model credentials remain on the FastAPI server."""

from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from research_qa.models import AskResponse, DocumentResponse, HealthResponse


class DemoAPIError(RuntimeError):
    """A user-facing error that never includes an upstream response body."""


class QAClient:
    def __init__(self, base_url: str, *, http_client: httpx.Client | None = None):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise DemoAPIError("QA_API_URL 必须是不含凭据或查询参数的 HTTP(S) 地址。")
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client

    def _request(self, method: str, endpoint: str, *, timeout: float, **kwargs) -> dict:
        try:
            if self.http_client is not None:
                response = self.http_client.request(
                    method, self.base_url + endpoint, timeout=timeout, **kwargs
                )
            else:
                # Short-lived clients avoid caching handles across browser sessions.
                with httpx.Client(trust_env=False) as client:
                    response = client.request(
                        method, self.base_url + endpoint, timeout=timeout, **kwargs
                    )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")
            return data
        except httpx.HTTPStatusError as exc:
            messages = {
                413: "上传超过服务端大小限制，请选择较小的 PDF。",
                415: "文件类型不受支持，请上传有效 PDF。",
                422: "文件或问题校验未通过。请检查 PDF 是否可读取、是否含文字，以及问题格式。",
                502: "模型服务暂时不可用，请检查后端配置后重试。",
            }
            message = messages.get(exc.response.status_code, "服务处理失败，请查看后端日志后重试。")
            raise DemoAPIError(message) from exc
        except httpx.TimeoutException as exc:
            raise DemoAPIError("请求超时。年报首次建库可能较慢，请稍后刷新后端状态。") from exc
        except httpx.HTTPError as exc:
            raise DemoAPIError("无法连接问答服务，请先启动 FastAPI 并检查 QA_API_URL。") from exc
        except (ValueError, TypeError) as exc:
            raise DemoAPIError("服务返回了无法识别的数据，请检查前后端版本。") from exc

    def health(self) -> HealthResponse:
        try:
            return HealthResponse.model_validate(self._request("GET", "/health", timeout=5))
        except ValidationError as exc:
            raise DemoAPIError("健康检查返回格式错误，请检查后端版本。") from exc

    def upload(self, name: str, data: bytes) -> DocumentResponse:
        try:
            result = self._request(
                "POST",
                "/documents",
                timeout=300,
                files={"file": (name, data, "application/pdf")},
            )
            return DocumentResponse.model_validate(result)
        except ValidationError as exc:
            raise DemoAPIError("上传结果格式错误，请检查后端版本。") from exc

    def ask(self, question: str, top_k: int) -> AskResponse:
        try:
            result = self._request(
                "POST", "/ask", timeout=180, json={"question": question, "top_k": top_k}
            )
            return AskResponse.model_validate(result)
        except ValidationError as exc:
            raise DemoAPIError("问答结果格式错误，请检查后端版本。") from exc
