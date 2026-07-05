.PHONY: run

run: ## Run the proxy with env vars from .env (if present)
	@( \
		if [ -f .env ]; then set -a; source .env; set +a; fi; \
		go run .; \
	)
