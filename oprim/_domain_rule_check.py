"""oprim.domain_rule_check — 单次通过领域特定规则校验静态代码.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: re (stdlib), 纯计算.

例:
    >>> code = "pub fn transfer(ctx: Context<Transfer>, amount: u64)"
    >>> result = domain_rule_check(code, domain="solana")
    >>> result["is_compliant"]
    False
"""

from __future__ import annotations

import re
from typing import Any


def domain_rule_check(
    code_snippet: str,
    *,
    domain: str = "solana",
) -> dict[str, Any]:
    """单次通过领域特定规则校验静态代码 (sol-advisor 机制).

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    支持的领域:
    - "solana": Anchor/Solana 程序安全检查
    - "ethereum": Solidity 安全检查
    - "security": 通用安全最佳实践
    - "rust": Rust 内存安全检查

    Args:
        code_snippet: 代码片段
        domain: 领域标识 ("solana" | "ethereum" | "security" | "rust")

    Returns:
        {
            "domain": str,
            "is_compliant": bool,
            "violations": [{"rule": str, "severity": "error"|"warning"}],
        }
    """
    violations: list[dict[str, str]] = []
    code = code_snippet

    if domain == "solana":
        # Anchor 安全检查
        if "AccountInfo" in code and "is_signer" not in code:
            violations.append(
                {
                    "rule": "Missing 'is_signer' check on AccountInfo parameter.",
                    "severity": "error",
                }
            )
        if "invoke_signed" in code and "reentrancy" not in code.lower():
            violations.append(
                {
                    "rule": "Potential reentrancy risk: no guard before CPI call 'invoke_signed'.",
                    "severity": "warning",
                }
            )
        if "transfer" in code and ".try_borrow_mut_lamports()" not in code:
            violations.append(
                {
                    "rule": "Lamport transfer should use try_borrow pattern for safety.",
                    "severity": "warning",
                }
            )
        if "unsafe" in code:
            violations.append(
                {
                    "rule": "Usage of 'unsafe' block detected in Solana program.",
                    "severity": "warning",
                }
            )
        if "panic!" in code:
            violations.append(
                {
                    "rule": "Avoid 'panic!' in on-chain programs — use require! or return Err.",
                    "severity": "error",
                }
            )

    elif domain == "ethereum":
        # Solidity 安全检查
        if ".transfer(" in code or ".send(" in code:
            violations.append(
                {
                    "rule": (
                        "Prefer .call{value: amount}('') over .transfer/.send "
                        "(gas limit changes)."
                    ),
                    "severity": "warning",
                }
            )
        if "tx.origin" in code:
            violations.append(
                {
                    "rule": "Avoid tx.origin for authentication — use msg.sender.",
                    "severity": "error",
                }
            )
        if "delegatecall" in code and "onlyOwner" not in code:
            violations.append(
                {
                    "rule": "delegatecall without access control — potential proxy vulnerability.",
                    "severity": "error",
                }
            )
        if "block.timestamp" in code and "require" not in code:
            violations.append(
                {
                    "rule": (
                        "block.timestamp used without validation — "
                        "susceptible to manipulation."
                    ),
                    "severity": "warning",
                }
            )

    elif domain == "rust":
        # Rust 安全检查
        if "unsafe" in code:
            violations.append(
                {
                    "rule": "Usage of 'unsafe' block — ensure Safety comment is present.",
                    "severity": "warning",
                }
            )
        if ".unwrap()" in code:
            violations.append(
                {
                    "rule": "Avoid .unwrap() in production code — use proper error handling.",
                    "severity": "warning",
                }
            )
        if "as " in code and any(t in code for t in ("u64", "i64", "u32", "i32", "usize", "isize")):
            violations.append(
                {
                    "rule": (
                        "Integer type conversion with 'as' may truncate silently — "
                        "consider .try_into()."
                    ),
                    "severity": "warning",
                }
            )

    elif domain == "security":
        # 通用安全检查
        if "password" in code.lower() and "=" in code:
            violations.append(
                {
                    "rule": "Possible hardcoded password detected.",
                    "severity": "error",
                }
            )
        if "exec(" in code or "eval(" in code:
            violations.append(
                {
                    "rule": "Avoid exec/eval with user-controlled input.",
                    "severity": "error",
                }
            )
        if re.search(
            r"SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*" + re.escape("' +"), code, re.IGNORECASE
        ):
            pass  # SQL injection pattern would need deeper analysis

    # SQL injection simple check
    if domain in ("security",) and (
        re.search(r"\+\s*['\"]", code) or re.search(r"f['\"].*\{.*\}.*SELECT", code, re.IGNORECASE)
    ):
        violations.append(
            {
                "rule": "Potential SQL injection: use parameterized queries.",
                "severity": "error",
            }
        )

    return {
        "domain": domain,
        "is_compliant": len(violations) == 0,
        "violations": violations,
    }
