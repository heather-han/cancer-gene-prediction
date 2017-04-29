data_directory='./data'
results_directory='./results'

sh process_data.sh $data_directory $results_directory
sh train_model.sh $data_directory $results_directory
sh test_model.sh $data_directory $results_directory
