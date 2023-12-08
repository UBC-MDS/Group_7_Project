
all: report/_build/html/bank_marketing_prediction.html

data/raw/bank_marketing_train.csv : scripts/import_data_split.py
	python scripts/import_data_split.py \
   	--data_id=222 \
   	--ratio=0.10

results/plots/class_imbalance.png : scripts/eda.py data/processed/bank_marketing_train.csv 
	python scripts/eda.py \
   	--train_path=data/processed/bank_marketing_train.csv \
   	--plot_path=results/plots \
   	--table_path=results/tables

results/models/preprocessor.pickle : scripts/preprocessor_script.py data/raw/bank_marketing_train.csv \
data/raw/bank_marketing_test.csv
	python scripts/preprocessor_script.py \
   	--raw_train_data=data/raw/bank_marketing_train.csv \
   	--raw_test_data=data/raw/bank_marketing_test.csv \
   	--output_preprocessed_data=data/processed/ \
   	--output_preprocessor=results/models/ \
   	--seed=522

# train KNN model, create visualize tuning, and save plot and model
results/tables/cv/knn_cv.csv : scripts/knn_model.py results/models/preprocessor.pickle \
data/raw/bank_marketing_train.csv
	python scripts/knn_model.py \
   	--preprocessor_path=results/models/preprocessor.pickle \
   	--train_data_path=data/raw/bank_marketing_train.csv \
   	--model_save_path=results/models/knn.pickle \
   	--table_dir=results/tables \
   	--plot_dir=results/plots

# train SVC model, create visualize tuning, and save plot and model
results/tables/cv/svc_cv.csv : scripts/svc_model.py results/models/preprocessor.pickle \
data/raw/bank_marketing_train.csv
	python scripts/svc_model.py \
   	--preprocessor_path=results/models/preprocessor.pickle \
   	--train_data_path=data/raw/bank_marketing_train.csv \
   	--model_save_path=results/models/svc.pickle \
   	--table_dir=results/tables \
   	--plot_dir=results/plots

# train logistic regression model, create visualize tuning, and save plot and model
results/tables/cv/lr_cv.csv : scripts/lr_model.py results/models/preprocessor.pickle \
data/raw/bank_marketing_train.csv
	python scripts/lr_model.py \
   	--preprocessor_path=results/models/preprocessor.pickle \
   	--train_data_path=data/raw/bank_marketing_train.csv \
   	--model_save_path=results/models/lr.pickle \
   	--table_dir=results/tables \
   	--plot_dir=results/plots

# train ramdom forest regression model, create visualize tuning, and save plot and model
results/tables/cv/rf_cv.csv : scripts/rf_model.py results/models/preprocessor.pickle \
data/raw/bank_marketing_train.csv
	python scripts/rf_model.py \
   	--preprocessor_path=results/models/preprocessor.pickle \
   	--train_data_path=data/raw/bank_marketing_train.csv \
   	--model_save_path=results/models/rf.pickle \
   	--table_dir=results/tables \
   	--plot_dir=results/plots

# perform model comparison and save results
results/tables/model_comparison.csv : scripts/model_compare.py results/tables/cv/knn_cv.csv \
results/tables/cv/svc_cv.csv results/tables/cv/lr_cv.csv results/tables/cv/rf_cv.csv
	python scripts/model_compare.py \
   	--result_folder_path=results/tables

# evaluate model on test data and save results
results/tables/best_model_confusion_matrix.csv: scripts/best_model_evaluation.py \
results/models/rf.pickle data/processed/bank_marketing_test.csv
	python scripts/best_model_evaluation.py \
   	--model_path=results/models/rf.pickle \
   	--test_data_path=data/processed/bank_marketing_test.csv \
   	--target_dir=results/tables

report/_build/html/bank_marketing_prediction.html: report/bank_marketing_prediction.ipynb \
results/plots/class_imbalance.png results/plots/numeric_cols.png \
results/tables/eda/data_summary.csv results/tables/eda/missing_values.csv \
results/tables/cv/knn_cv.csv results/plots/knn.png results/tables/knn.csv \
results/tables/cv/svc_cv.csv results/plots/svc.png results/tables/svc.csv \
results/tables/cv/lr_cv.csv results/plots/lr.png results/tables/lr.csv \
results/tables/cv/rf_cv.csv results/plots/rf.png results/tables/rf.csv \
results/tables/model_comparison.csv results/tables/best_model_confusion_matrix.csv \
results/tables/best_model_score.csv
	jupyter-book build report
	cp -r report/_build/html/* docs

clean : 
	rm -rf data/raw/bank_marketing_train.csv data/raw/bank_marketing_test.csv
	rm -rf results/plots/class_imbalance.png results/plots/numeric_cols.png \
	results/tables/eda/data_summary.csv results/tables/eda/missing_values.csv
	rm -rf results/models/preprocessor.pickle data/processed/preprocessed_X_test.csv \
	data/processed/preprocessed_X_train.csv
	rm -rf results/models/knn.pickle results/plots/knn.png results/tables/cv/knn_cv.csv \
	results/tables/knn.csv
	rm -rf results/models/svc.pickle results/plots/svc.png results/tables/cv/svc_cv.csv \
	results/tables/svc.csv
	rm -rf results/models/lr.pickle results/plots/lr.png results/tables/cv/lr_cv.csv \
	results/tables/lr.csv
	rm -rf results/models/rf.pickle results/plots/rf.png results/tables/cv/rf_cv.csv \
	results/tables/rf.csv
	rm -rf results/tables/model_comparison.csv
	rm -rf results/tables/best_model_confusion_matrix.csv results/tables/best_model_score.csv
	rm -rf report/_build