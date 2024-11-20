search_dir=../humaneval_data/
for entry in "$search_dir"*.py
do
  temp=${entry#$search_dir}
  module_name=${temp%.*}
  echo "Start: $module_name"
  timeout 30 bash -c -- "mut.py --target ${module_name} --unit-test test_${module_name} --runner pytest" >> mutation_humaneval_3.txt
  echo "Done: $module_name"
done