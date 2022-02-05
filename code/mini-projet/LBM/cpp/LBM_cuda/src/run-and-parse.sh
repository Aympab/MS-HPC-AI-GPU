#!/bin/bash

csvfile=log/perfGPU.csv
inifile=flowAroundCylinder.ini


echo "Running!"

mkdir -p log
rm $csvfile

echo "size;nbIte;time;bandwith;prop;gflop" > $csvfile

array=( 256 512 1024 2048 4096 8192 16384 ) # 32768 65536 ) #nx
array2=( 64 128 256 512 1024 2048 4096 ) # 8192 16384) #ny

for i in "${!array[@]}"; do
  nx=${array[i]}
  ny=${array2[i]}

  echo "Loop $i... nx=$nx ny=$ny"

  sed -i "11s/.*/nx=$nx/" inifile #nx is on the 11th line
  sed -i "12s/.*/ny=$ny/" inifile #ny is on the 12th line

  ./lbmFlowAroundCylinder > tmp_log

  line=$(grep to_parse tmp_log)

  size=$(echo $line | cut -d ';' -f2)
  nbIte=$(echo $line | cut -d ';' -f3)
  time=$(echo $line | cut -d ';' -f4)
  bandwith=$(echo $line | cut -d ';' -f5)
  prop=$(echo $line | cut -d ';' -f6)
  gflop=$(echo $line | cut -d ';' -f7)

  echo "${size};${nbIte};${time};${bandwith};${prop};${gflop}" >> $csvfile
done

rm tmp_log

echo "Done !"
