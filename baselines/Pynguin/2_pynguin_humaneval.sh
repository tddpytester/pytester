search_dir=./humaneval_data/
for entry in "$search_dir"*.py
do
  temp=${entry#$search_dir}
  module_name=${temp%.*}
  echo "Start: $module_name"
  timeout 60 bash -c -- "pynguin --project-path ./humaneval_data --output-path ./humaneval_test_3 --report-dir ./humaneval_test_3 --module-name ${module_name} --max_size 3" # -v
  echo "Done: $module_name"
done
