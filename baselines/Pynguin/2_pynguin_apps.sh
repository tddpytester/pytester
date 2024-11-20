search_dir=./apps_data/
for entry in "$search_dir"*.py
do
  temp=${entry#$search_dir}
  module_name=${temp%.*}
  echo "Start: $module_name"
  timeout 5 bash -c -- "pynguin --project-path ./apps_data --output-path ./apps_test --report-dir ./apps_test --module-name ${module_name}" # -v
  echo "Done: $module_name"
done