.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = ntu-rgb-d
PYTHON_INTERPRETER = python3

# Make Features variables
DATA_PATH = "/home/gnocchi/NTU/"
NTU_RGBD_DATA_PATH = "/media/gnocchi/Seagate Backup Plus Drive/NTU-RGB-D/"
PROCESSED_DATA :=$(PROJECT_DIR)/data/processed/
DATASET_TYPE = "" # [SKELETON, RGB, IR]
COMPRESSION = ""
COMPRESSION_OPTS = 9

# Make Train variables
MODEL_FOLDER :=$(PROJECT_DIR)/models/
EVALUATION_TYPE=cross_subject
MODEL_TYPE=FUSION
USE_POSE=False
USE_IR=False
PRETRAINED=False
USE_CROPPED_IR=False
OPTIMIZER=ADAM
LEARNING_RATE=1e-4
WEIGHT_DECAY=0
GRADIENT_THRESHOLD=0
EPOCHS=30
BATCH_SIZE=8
ACCUMULATION_STEPS=1
SUB_SEQUENCE_LENGTH=20 
NORMALIZE_SKELETON=True
NORMALIZATION_TYPE=1-COORD-SYS
KINEMATIC_CHAIN_SKELETON=False
AUGMENT_DATA=True
USE_VALIDATION=True
EVALUATE_TEST=True

# Plot confusion matrix variables
MODEL_FILE=""

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Make Features
features: 
	$(PYTHON_INTERPRETER) src/features/build_features.py \
	--data_path=$(NTU_RGBD_DATA_PATH) \
	--output_folder=$(NTU_RGBD_DATA_PATH) \
	--dataset_type=$(DATASET_TYPE) \
	--compression=$(COMPRESSION) \
	--compression_opts=$(COMPRESSION_OPTS)

## Make Train
train: 
	$(PYTHON_INTERPRETER) src/models/train_model.py --data_path=$(NTU_RGBD_DATA_PATH) \
	--output_folder=$(MODEL_FOLDER) \
	--evaluation_type=$(EVALUATION_TYPE) \
	--model_type=$(MODEL_TYPE) \
	--use_pose=$(USE_POSE) \
	--use_ir=$(USE_IR) \
	--pretrained=$(PRETRAINED) \
	--use_cropped_IR=$(USE_CROPPED_IR) \
	--optimizer=$(OPTIMIZER) \
	--learning_rate=$(LEARNING_RATE) \
	--weight_decay=$(WEIGHT_DECAY) \
	--gradient_threshold=$(GRADIENT_THRESHOLD) \
	--epochs=$(EPOCHS) \
	--batch_size=$(BATCH_SIZE) \
	--accumulation_steps=$(ACCUMULATION_STEPS) \
	--sub_sequence_length=$(SUB_SEQUENCE_LENGTH) \
	--normalize_skeleton=$(NORMALIZE_SKELETON) \
	--normalization_type=$(NORMALIZATION_TYPE) \
	--kinematic_chain_skeleton=$(KINEMATIC_CHAIN_SKELETON) \
	--augment_data=$(AUGMENT_DATA) \
	--use_validation=$(USE_VALIDATION) \
	--evaluate_test=$(EVALUATE_TEST)

## Make Visualize
confusion_matrix:
	$(PYTHON_INTERPRETER) src/models/plot_confusion_matrix.py --data_path=$(NTU_RGBD_DATA_PATH) \
	--model_folder=$(MODEL_FOLDER) \
	--model_file=$(MODEL_FILE) \
	--evaluation_type=$(EVALUATION_TYPE) \
	--model_type=$(MODEL_TYPE) \
	--batch_size=$(BATCH_SIZE) \
	--sub_sequence_length=$(SUB_SEQUENCE_LENGTH) \
	--normalize_skeleton=$(NORMALIZE_SKELETON) \
	--normalization_type=$(NORMALIZATION_TYPE) \
	--kinematic_chain_skeleton=$(KINEMATIC_CHAIN_SKELETON)

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
