# Account Recovery Implementation Plan

**Goal:** 在没有业务请求时恢复到期额度，并使用仍有效的关联 Web SSO 恢复失效 Build 凭据；保留账号运营状态和最新凭据。

**Architecture:** 统一后台调度和总探测预算，复用 Build 业务选号/额度状态机以及 Web、Console 的额度窗口队列。凭据恢复使用关联身份核对、与续期共用的锁以及按 ID 的条件更新；不通过导入或全池模型测试恢复账号。配置文件新增独立且默认关闭的入口，无需前端改动。

**Tech Stack:** Go、既有 Provider adapters、SQLite/PostgreSQL、既有内存/Redis 租约；不增加依赖。

## 已确认范围

- 用户最新要求“尽可能恢复”扩展原范围，允许有效 SSO 重新授权失效 Build；额度恢复不等同认证恢复。
- 三类 Provider 统一调度。扫描和维护可以包含禁用账号；只有经迁移记录核对、排除人工停用并登记在恢复名单内的 Build，凭据与额度都恢复后才自动启用。用户已明确同意该规则；人工操作 Enabled 会撤销名单授权。
- 只探测到期耗尽账号，不进行全池模型检测；SSO 恢复只处理已有、身份确定的关联 Build。
- 普通 Token 续期、手动 SSO 转换与后台重新授权必须共用 Build 凭据锁。
- 暂时失败持久退避；明确拒绝 SSO 不再反复调用。任何日志不得包含凭据或上游原始敏感响应。
- 生产初始周期 5m、总批量 10、并发 1；SSO 重授权默认子上限 1，生产首轮验证后可以在总批量内调整。首次立即运行，后续在本轮结束后等待 5m。
- 252 上备份 SQLite 和配置，固定镜像；回滚只退代码/配置，不覆盖运行后更新的账号库；不复制 taiyi，不启动已退役 Grok CLI。

## Task 1: 配置与运行边界

Files: backend/internal/infra/config/config.go, account_recovery_test.go, config.example.yaml。

先写加载/默认关闭/边界非法配置测试，运行失败后增加配置。周期、批量、并发、探测超时、Provider 开关、包含禁用账号、SSO 开关/单轮限额/退避均可配置。配置变更重启生效。

## Task 2: Build 探测与持久筛选

Files: backend/internal/application/gateway/account_recovery.go, corresponding tests; backend/internal/infra/persistence/relational/account_recovery.go and tests。

先测试到期筛选、禁用策略、认证/冷却/模型/出口排除和并发领取。复用 SQL ClaimQuotaProbe 及业务账号并发；限定一个已启用模型的最小请求。成功必须等待流终态和非空输出，失败保留原状态机与退避。取消和陈旧租约不得恢复账号。

## Task 3: SSO 认证恢复

Files: backend/internal/application/account/reauth_recovery.go and tests; relational token CAS helper and tests。

先测试关联身份不符、并发续期、人工禁用保留、凭据条件更新、网络超时、明确认证拒绝和中断。新凭据仅在身份/版本检查通过后写回同一账号；不使用 persistSeed 的导入语义。持久记录尝试租约和下次时间，防重启重复重授权。

## Task 4: 统一调度及 Web/Console 整合

Files: backend/internal/application/accountrecovery/, backend/internal/application/quotarecovery/, backend/internal/app/application.go。

先测试全局批量/并发、轮次不重叠、Provider 过滤、错误退避、队列里已被禁用/失效的账号以及服务中断。启用统一调度后由它驱动原额度队列，避免原周期任务重复运行。为恢复统计输出 scanned/claimed/recovered/failed/skipped。

## Task 5: 验证、评审与部署

- 运行新增定向测试和 race；全后端 go test ./... 与 go vet ./...；前端无变更但按构建门禁完成 lint/build。
- 独立代码评审，修复结论并重新验证。
- 在 252 创建权限 700 的私有备份目录，保存一致性 SQLite 和原配置、镜像信息、恢复操作说明；不下载凭据备份。
- 构建固定 revision 镜像，先默认关闭验证，再保守启用；验证 readyz、grok-4.5/grok-4.6 流式终态与非空输出，以及 New API 渠道 103，确保 104 仍禁用。
- 记录实际恢复统计与暂不能恢复原因，备份位置、固定镜像、最新数据兼容回滚步骤；不宣称未完成的全部账号恢复。
