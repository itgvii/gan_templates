IMAGE_NAME = skins_gan:1.0
ARCHIVE_NAME = skins_gan.tar.gz
VPS_USER = root
VPS_HOST = 5.129.207.153
REMOTE_DIR = /root
CONTAINER_NAME = skins_gan

test_build:
	@echo "[1/4] Прогон тестов и сборка"
	poetry run pytest tests || exit 1

	docker build -t $(IMAGE_NAME) .

	docker run --rm -d --name $(CONTAINER_NAME) -p 8000:8000 $(IMAGE_NAME)
	@sleep 5

	@echo "Тестируем контейнер"
	@poetry run python tests_docker/test_download.py; \
		STATUS=$$?; \
		docker stop $(CONTAINER_NAME); \
		if [ $$STATUS -ne 0 ]; then \
			echo "Контейнер не прошел тесты"; \
			exit $$STATUS; \
		else \
			docker rm $(CONTAINER_NAME); \
			echo "Контейнер прошел тесты"; \
		fi

package:
	@echo "[2/4] Упаковка Docker-образа"
	docker save $(IMAGE_NAME) | gzip > $(ARCHIVE_NAME)

push:
	@echo "[3/4] Отправка архива на VPS"
	scp $(ARCHIVE_NAME) $(VPS_USER)@$(VPS_HOST):$(REMOTE_DIR)/

deploy:
	@echo "[4/4] Деплой на VPS"
	ssh $(VPS_USER)@$(VPS_HOST) '\
		cd $(REMOTE_DIR) && \
		gunzip -f $(ARCHIVE_NAME) && \
		docker load -i $(ARCHIVE_NAME:.gz=) && \
		if [ $$(docker ps -q -f name=$(CONTAINER_NAME)) ]; then \
			docker stop $(CONTAINER_NAME); \
		else \
			echo "Контейнер $(CONTAINER_NAME) не запущен, останавливать не нужно"; \
		fi && \
		docker rm -f $(CONTAINER_NAME) || true && \
		docker run --rm -d --name $(CONTAINER_NAME) -p 8000:8000 $(IMAGE_NAME) \
	'

clean:
	rm -f $(ARCHIVE_NAME)

all: test_build package push deploy clean
	@echo "✅ Деплой завершён"