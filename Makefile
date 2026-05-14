.DEFAULT_GOAL := help

SSH_HOST ?= 209.182.232.155
SSH_USER ?= root
SSH_PORT ?= 22

REMOTE_APP_DIR ?= /opt/ar_app_backend
REMOTE_MODEL_DIR ?= $(REMOTE_APP_DIR)/model
REMOTE_MODEL_FILE ?= $(REMOTE_MODEL_DIR)/final_model.pt

MODEL_FILE ?= model/final_model.pt

REMOTE := $(SSH_USER)@$(SSH_HOST)

.PHONY: help ssh ensure-remote-model-dir copy-model deploy-model

help:
	@printf '%s\n' \
		'Make targets:' \
		'  make ssh                     Connect to the VM over SSH' \
		'  make ensure-remote-model-dir Create the remote models directory' \
		'  make copy-model              Copy $(MODEL_FILE) to $(REMOTE_MODEL_FILE)' \
		'  make deploy-model            Create the remote dir and copy the model'

ssh:
	ssh -p $(SSH_PORT) $(REMOTE)

ensure-remote-model-dir:
	ssh -p $(SSH_PORT) $(REMOTE) "mkdir -p '$(REMOTE_MODEL_DIR)'"

copy-model: ensure-remote-model-dir
	scp -P $(SSH_PORT) "$(MODEL_FILE)" "$(REMOTE):$(REMOTE_MODEL_FILE)"

deploy-model: copy-model