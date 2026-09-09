# 后台账号恢复巡检

`accountRecovery.enabled` 默认关闭，修改配置后重启生效。开启后统一驱动 Build 到期额度探测、Web/Console 原额度窗口队列，以及可选的关联 SSO 重授权；原 Web/Console 定时补查不再重复启动。普通 OAuth Token 续期继续复用原服务。

```yaml
accountRecovery:
  enabled: false
  interval: 5m
  batchSize: 10
  concurrency: 1
  probeTimeout: 90s
  build: true
  web: true
  console: true
  includeDisabled: false
  ssoReauth: false
  reauthBatchSize: 1
  reauthBackoffBase: 1h
  reauthBackoffMax: 24h
  buildModels: [grok-4.5, grok-4.6]
```

首次立即运行，随后每轮完成后等待 `interval`。每轮共用 `batchSize` 领取预算，SSO 尝试另受 `reauthBatchSize` 约束；扫描数可以大于领取数。阶段顺序轮换，单机运行标志和分布式租约阻止轮次重叠，每个账号仍使用业务侧相同的额度、凭据或额度窗口锁。

Build 只选择一个允许且可路由的模型，遵守账号冷却、模型限制、BotFlag 策略、出口和并发上限。免费额度必须收到有效流终态及非空内容才算恢复；付费额度使用既有账单探测。所有完成和失败写入都校验领取租约，旧请求无法覆盖后来的状态。

`includeDisabled` 允许维护禁用账号，本身不授予业务启用权限。Web/Console 使用现有会话查询到期额度，不主动续签已经失效的 SSO。`ssoReauth` 允许有效关联 Web SSO 为同一身份的 Build 重新授权；SSO 本身被明确拒绝时需要补充有效凭据，网络错误按退避重试。新凭据必须连同 client ID 原子保存；取消也不能丢失已成功轮换的 Token。

## 经授权的迁移账号自动恢复业务

`account_recovery_activations` 是显式授权名单，不会按禁用状态或来源前缀自动填充。导入前须核对原迁移映射、当前身份与来源，排除明确人工停用记录，并将审核清单和数据库备份保存在部署服务器。名单行绑定 account ID、source key、user ID 和 email；普通运维不应批量写入该表。

名单内账号凭据恢复后仍保持禁用，准备到期额度检查。只有额度探测成功的租约事务才能启用并消费授权；无名单的账号保持原 Enabled。管理员显式修改 Enabled（包括再次设为 false）先撤销授权，后续后台结果无法覆盖人工决定。

## 日志与回滚

`account_recovery_patrol` 记录 `scanned / claimed / recovered / failed / skipped`、额度恢复数、凭据恢复数及是否中断。`account_recovery_result` 只记录账号 ID、类型、结果和固定原因码。恢复数不等同自动启用数，不输出凭据或原始上游响应。

上线前使用 SQLite 在线备份接口保存一致性快照，并保存配置、加密密钥所在配置和原镜像 ID；目录 0700、文件 0600。先用新镜像保持巡检关闭验证，再启用保守参数。

回滚前再备份最新数据库，停服务后恢复旧镜像和配置，挂载继续使用当前数据卷。**不能直接用上线前旧数据库覆盖当前库**，否则会丢失已经轮换的 Token。新增授权表可保留，旧版本忽略；禁止删除数据卷或清理回滚镜像。
