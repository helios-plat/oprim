"""oprim 统一异常体系."""

from __future__ import annotations


class OprimError(Exception):
    """所有 oprim 失败时抛出的基类异常."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        s = super().__str__()
        if self.cause:
            s += f" (caused by: {self.cause})"
        return s


class FileOprimError(OprimError):
    """文件 IO 类 oprim 失败."""


class GitOprimError(OprimError):
    """Git subprocess 类 oprim 失败."""


class ShellOprimError(OprimError):
    """bash_exec 类 oprim 失败."""


class ParseOprimError(OprimError):
    """解析类 oprim 失败."""


class PathSecurityError(OprimError):
    """路径安全校验失败（沙箱越界 / 路径穿越）."""


# Legacy/Existing exceptions kept for backward compatibility (though no current usage found)
class OprimConnectionError(OprimError):
    """External connection failure (docker daemon / pg / rabbitmq / http / etc.)."""


class OprimTimeoutError(OprimError):
    """Operation timed out."""


class OprimNotFoundError(OprimError):
    """Target object not found (container_id missing, queue absent, etc.)."""


class OprimAuthError(OprimError):
    """Authentication failure (S3 / DB / Caddy admin / etc.)."""


class OprimValidationError(OprimError):
    """Input parameter validation failed."""


class LLMOprimError(OprimError):
    """LLM 调用失败（含 provider 错误、预算超出、格式错误）."""


class BudgetExceededError(LLMOprimError):
    """Token 预算超出."""


class PromptOprimError(OprimError):
    """Prompt 构建或消息处理失败."""


class SearchOprimError(OprimError):
    """Web 搜索调用失败."""


class HttpOprimError(OprimError):
    """HTTP 请求失败."""


class SnapshotOprimError(OprimError):
    """会话快照持久化失败."""


class PaymentOprimError(OprimError):
    """支付 provider 调用失败(ext_pay_* 原子)."""


class ShippingOprimError(OprimError):
    """物流履约 provider 调用失败(ext_ship_* 原子)."""


class SearchIndexOprimError(OprimError):
    """搜索索引 provider 调用失败(ext_search_* 原子,区别于网页搜索的 SearchOprimError)."""


class NotifyOprimError(OprimError):
    """通知 provider 调用失败(ext_notify_send 原子)."""


class TaxOprimError(OprimError):
    """税费计算 provider 调用失败(ext_tax_calculate 原子)."""
