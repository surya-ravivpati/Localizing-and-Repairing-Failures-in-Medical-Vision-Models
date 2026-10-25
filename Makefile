# RadLLM prompt-eval pipeline. Run from pipeline/.
PY = python3

.PHONY: all dataset eda infer eval analyze figures test clean

all: dataset infer eval analyze figures

dataset:      ## build labeled studies + locked balanced test/dev manifests
	$(PY) scripts/01_build_dataset.py

eda:          ## print + save distribution / ambiguity signal summaries
	$(PY) scripts/02_eda.py

infer:        ## run 5 prompts over the test set (backend from config.yaml)
	$(PY) scripts/03_run_inference.py --split test

eval:         ## accuracy, groundedness, hallucination, calibration, judge
	$(PY) scripts/04_evaluate.py

analyze:      ## paired stats, bootstrap, GLMM, H1-H4 tests
	$(PY) scripts/05_analyze.py

figures:      ## reliability / error-profile / interaction / dissociation PNGs
	$(PY) scripts/06_figures.py

test:         ## fast unit tests on the deterministic core
	$(PY) tests/test_pipeline.py

clean:        ## remove generated results (keeps manifests)
	rm -f results/predictions_*.csv results/*_test.csv results/summary_*.csv results/stats_*.csv
	rm -rf results/figures
