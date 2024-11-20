search_dir=./mbpp_data/
for entry in "$search_dir"*.py
do
  temp=${entry#$search_dir}
  module_name=${temp%.*}
  echo "Start: $module_name"
  timeout 60 bash -c -- "pynguin --project-path ./mbpp_data --output-path ./mbpp_test_5 --report-dir ./mbpp_test_5 --module-name ${module_name} --max_size 5" # -v
  echo "Done: $module_name"
done