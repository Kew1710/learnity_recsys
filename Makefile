.PHONY: up down logs migrate migrate-create seed dev stop dev-logs dev-status test lint play stats install restart-retrieval restart-retrieval-baseline ab ab-quick backend-up backend-migrate backend-run backend-down

PYTHON  = .venv/bin/python
PIP     = .venv/bin/pip
UVICORN = .venv/bin/uvicorn

# ─── Инфраструктура (PostgreSQL, Neo4j, Kafka) ────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

# ─── Зависимости ──────────────────────────────────────────────────────────────

install:
	$(PIP) install -r requirements-shared.txt
	$(PIP) install pytest pytest-asyncio rich respx

# ─── Миграции / Seed ──────────────────────────────────────────────────────────

migrate:
	$(PYTHON) -m alembic upgrade head

migrate-create:
	$(PYTHON) -m alembic revision --autogenerate -m "$(name)"

seed:
	@echo "→ Seed граф знаний (Neo4j)..."
	NEO4J_URI=bolt://127.0.0.1:7688 $(PYTHON) -m services.graph.seed
	@echo "→ Инвалидация кэша графа..."
	curl -s -X POST http://127.0.0.1:8002/cache/invalidate | $(PYTHON) -m json.tool || true
	@echo "→ Seed банк заданий (PostgreSQL)..."
	$(PYTHON) -m services.task_bank.seed

# ─── Запуск сервисов (локально, без Docker) ───────────────────────────────────

dev:
	@mkdir -p .pids logs
	@echo "→ Запускаем сервисы..."
	@NEO4J_URI=bolt://localhost:7688 \
	 $(UVICORN) services.profile.main:app   --port 8001 --reload --reload-dir services >logs/profile.log   2>&1 & echo $$! > .pids/profile.pid
	@NEO4J_URI=bolt://localhost:7688 \
	 $(UVICORN) services.graph.main:app     --port 8002 --reload --reload-dir services >logs/graph.log     2>&1 & echo $$! > .pids/graph.pid
	@$(UVICORN) services.task_bank.main:app --port 8003 --reload --reload-dir services >logs/task_bank.log 2>&1 & echo $$! > .pids/task_bank.pid
	@PROFILE_URL=http://localhost:8001 GRAPH_URL=http://localhost:8002 TASK_BANK_URL=http://localhost:8003 \
	 $(UVICORN) services.retrieval.main:app --port 8004 --reload --reload-dir services >logs/retrieval.log 2>&1 & echo $$! > .pids/retrieval.pid
	@PROFILE_URL=http://localhost:8001 RETRIEVAL_URL=http://localhost:8004 \
	 $(UVICORN) services.gateway.main:app   --port 8005 --reload --reload-dir services >logs/gateway.log   2>&1 & echo $$! > .pids/gateway.pid
	@PROFILE_URL=http://localhost:8001 GRAPH_URL=http://localhost:8002 \
	 $(UVICORN) services.macro.main:app     --port 8006 --reload --reload-dir services >logs/macro.log     2>&1 & echo $$! > .pids/macro.pid
	@sleep 2
	@echo ""
	@echo "✅ Сервисы запущены:"
	@echo "   Gateway   → http://localhost:8005"
	@echo "   Profile   → http://localhost:8001"
	@echo "   Graph     → http://localhost:8002"
	@echo "   TaskBank  → http://localhost:8003"
	@echo "   Retrieval → http://localhost:8004"
	@echo "   Macro     → http://localhost:8006"
	@echo ""
	@echo "Логи: make dev-logs  |  Остановить: make stop"

stop:
	@echo "→ Останавливаем сервисы..."
	@for f in .pids/*.pid; do \
	  [ -f "$$f" ] && kill $$(cat $$f) 2>/dev/null && rm "$$f" || true; \
	done
	@echo "✅ Готово"

# A/B: перезапуск retrieval в режиме LinUCB (ENABLE_CONTROL_GROUP=false, по умолчанию)
restart-retrieval:
	@[ -f .pids/retrieval.pid ] && kill $$(cat .pids/retrieval.pid) 2>/dev/null || true
	@sleep 1
	@PROFILE_URL=http://localhost:8001 GRAPH_URL=http://localhost:8002 TASK_BANK_URL=http://localhost:8003 \
	 ENABLE_CONTROL_GROUP=false \
	 $(UVICORN) services.retrieval.main:app --port 8004 --reload >logs/retrieval.log 2>&1 & echo $$! > .pids/retrieval.pid
	@sleep 1
	@echo "✅ Retrieval перезапущен (LinUCB, ENABLE_CONTROL_GROUP=false)"

# A/B: перезапуск retrieval в режиме Baseline (ENABLE_CONTROL_GROUP=true)
restart-retrieval-baseline:
	@[ -f .pids/retrieval.pid ] && kill $$(cat .pids/retrieval.pid) 2>/dev/null || true
	@sleep 1
	@PROFILE_URL=http://localhost:8001 GRAPH_URL=http://localhost:8002 TASK_BANK_URL=http://localhost:8003 \
	 ENABLE_CONTROL_GROUP=true \
	 $(UVICORN) services.retrieval.main:app --port 8004 --reload >logs/retrieval.log 2>&1 & echo $$! > .pids/retrieval.pid
	@sleep 1
	@echo "✅ Retrieval перезапущен (Baseline, ENABLE_CONTROL_GROUP=true)"

dev-logs:
	@tail -f logs/*.log

dev-status:
	@echo "Gateway   (8005):" && curl -s http://127.0.0.1:8005/health | $(PYTHON) -m json.tool || echo "  не отвечает"
	@echo "Profile   (8001):" && curl -s http://127.0.0.1:8001/health | $(PYTHON) -m json.tool || echo "  не отвечает"
	@echo "Graph     (8002):" && curl -s http://127.0.0.1:8002/health | $(PYTHON) -m json.tool || echo "  не отвечает"
	@echo "TaskBank  (8003):" && curl -s http://127.0.0.1:8003/health | $(PYTHON) -m json.tool || echo "  не отвечает"
	@echo "Retrieval (8004):" && curl -s http://127.0.0.1:8004/health | $(PYTHON) -m json.tool || echo "  не отвечает"
	@echo "Macro     (8006):" && curl -s http://127.0.0.1:8006/health | $(PYTHON) -m json.tool || echo "  не отвечает"

# ─── Инструменты ──────────────────────────────────────────────────────────────

play:
	$(PYTHON) tools/play.py

stats:
	$(PYTHON) tools/stats.py

sandbox:
	$(PYTHON) tools/sandbox.py

# A/B тест: make ab | make ab SEED=7 N=10 TASKS=100 | make ab-quick
ab:
	$(PYTHON) tools/run_ab.py --seed $(or $(SEED),42) --n $(or $(N),30) --tasks $(or $(TASKS),250)

ab-quick:
	$(PYTHON) tools/run_ab.py --seed 42 --n 5 --tasks 50 --warmup-n 5 --warmup-tasks 30

# Только LinUCB:
#   make linucb
#   make linucb N=5 TASKS=200 WARMUP=0
#   make linucb PLAN_MODE=target_mastery PLAN_KC=kc_quadratic_eq
#   make linucb PLAN_MODE=coverage PLAN_VARIANT=frontier PLAN_BUDGET=200
linucb:
	$(PYTHON) tools/run_ab.py --variant linucb --seed $(or $(SEED),42) --n $(or $(N),30) --tasks $(or $(TASKS),250) \
		$(if $(filter 0,$(WARMUP)),--skip-warmup,) \
		$(if $(PLAN_MODE),--plan-mode $(PLAN_MODE),) \
		$(if $(PLAN_KC),--plan-kc $(PLAN_KC),) \
		$(if $(PLAN_VARIANT),--plan-variant $(PLAN_VARIANT),) \
		$(if $(PLAN_BUDGET),--plan-budget $(PLAN_BUDGET),)

# ─── Backend (Go) интеграция ──────────────────────────────────────────────────

BACKEND_DIR    = backend/learnity
BACKEND_PG_DSN = "host=localhost port=5433 user=postgres password=postgres dbname=postgres sslmode=disable"
GOOSE          = $(HOME)/go/bin/goose

backend-up:
	@echo "→ Поднимаем backend инфраструктуру (PG:5433 + Redis:6379)..."
	docker compose -f docker-compose.backend.yml up -d
	@echo "→ Ждём готовности PostgreSQL..."
	@until docker exec $$(docker compose -f docker-compose.backend.yml ps -q backend-postgres) pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
	@echo "✅ Backend инфраструктура готова"

backend-migrate: backend-up
	@echo "→ Запускаем миграции через psql..."
	@$(PYTHON) -c "\
import os, re; \
d='$(BACKEND_DIR)/migrations'; \
files=sorted(f for f in os.listdir(d) if f.endswith('.sql')); \
parts=[re.search(r'-- \+goose Up\n(.*?)(?:-- \+goose Down|\Z)', open(os.path.join(d,f)).read(), re.DOTALL) for f in files]; \
sql='\n\n'.join(m.group(1).strip() for m in parts if m and m.group(1).strip()); \
open('/tmp/_backend_migrations.sql','w').write(sql)"
	@docker exec -i learnity-backend-postgres-1 psql -U postgres -d postgres < /tmp/_backend_migrations.sql
	@echo "✅ Миграции применены"

backend-run:
	@echo "→ Запускаем backend (Go) на :8080..."
	@echo "   Swagger: http://localhost:8080/swagger/index.html"
	cd $(BACKEND_DIR) && env $$(grep -v '^#' .env.local.integration | grep -v '^$$' | \
	  sed 's/POSTGRES_CONN=.*/POSTGRES_CONN=postgres:\/\/postgres:postgres@localhost:5433\/postgres?sslmode=disable/' | \
	  xargs) go run cmd/main.go

backend-down:
	docker compose -f docker-compose.backend.yml down

# ─── Тесты / Lint ─────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest services/ shared/ -v

lint:
	$(PYTHON) -m ruff check shared/ services/
