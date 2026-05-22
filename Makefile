# RAG Toolkit — operator Makefile (GPU benchmarking + HF auth)
#
# Usage:
#   make gpu-up      # build images + start stack incl. benchmark-runner (GPU)
#   make hf-verify   # verify the HuggingFace token authenticates
#   make smoke       # verify the runner sees the GPU
#   make bench       # run full GPU experiment + benchmark pipeline
#   make report      # rebuild REPORT.md + AGGREGATE_REPORT.md only
#   make shell       # interactive shell inside the runner
#   make logs        # follow stack logs
#   make ps          # show stack status
#   make down        # stop + remove the stack
#   make gpu-bench   # gpu-up + hf-verify + smoke + bench in one go

COMPOSE := docker compose -f docker-compose.yml -f docker-compose.gpu.yml
RUNNER  := rag-benchmark-runner

.PHONY: gpu-up hf-verify smoke bench report shell logs ps down gpu-bench

gpu-up:
	$(COMPOSE) up -d --build

hf-verify:
	docker exec -i $(RUNNER) bash /opt/ops/hf-verify.sh

smoke:
	docker exec -i $(RUNNER) bash /opt/ops/gpu-smoke.sh

bench:
	docker exec -i $(RUNNER) bash /opt/ops/run-benchmarks.sh

report:
	docker exec -i $(RUNNER) bash /opt/ops/build-report.sh

shell:
	docker exec -it $(RUNNER) bash

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

down:
	$(COMPOSE) down

gpu-bench: gpu-up hf-verify smoke bench
