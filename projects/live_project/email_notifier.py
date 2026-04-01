from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.header import Header
from email.message import EmailMessage
from typing import Iterable


class EmailNotifyError(RuntimeError):
    pass


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    sender: str
    recipients: list[str]
    use_ssl: bool = True
    use_starttls: bool = False
    subject_prefix: str = "[AutoTrade]"


class EmailNotifier:
    def __init__(self, config: EmailConfig):
        self.config = config

    @staticmethod
    def _normalize_recipients(recipients: Iterable[str]) -> list[str]:
        out: list[str] = []
        for r in recipients:
            text = str(r).strip()
            if text and text not in out:
                out.append(text)
        return out

    def send(self, subject: str, body: str) -> None:
        recipients = self._normalize_recipients(self.config.recipients)
        if not recipients:
            raise EmailNotifyError("收件人为空")

        full_subject = f"{self.config.subject_prefix} {subject}".strip()
        msg = EmailMessage()
        msg.set_content(body, subtype="plain", charset="utf-8")
        # 显式按 UTF-8 编码主题，避免部分邮箱网关错误降级为 '?'
        msg["Subject"] = str(Header(full_subject, "utf-8"))
        msg["From"] = self.config.sender
        msg["To"] = ", ".join(recipients)

        try:
            if self.config.use_ssl:
                with smtplib.SMTP_SSL(self.config.smtp_host, int(self.config.smtp_port), timeout=15) as server:
                    if self.config.smtp_user:
                        server.login(self.config.smtp_user, self.config.smtp_password)
                    server.sendmail(self.config.sender, recipients, msg.as_string())
            else:
                with smtplib.SMTP(self.config.smtp_host, int(self.config.smtp_port), timeout=15) as server:
                    server.ehlo()
                    if self.config.use_starttls:
                        server.starttls()
                        server.ehlo()
                    if self.config.smtp_user:
                        server.login(self.config.smtp_user, self.config.smtp_password)
                    server.sendmail(self.config.sender, recipients, msg.as_string())
        except Exception as exc:  # noqa: BLE001
            raise EmailNotifyError(f"邮件发送失败: {exc}") from exc
