
i=1
for f in *train*.dat
do

  if [ $i = 1 ]; then
     tail -n +2 $f | cut -f 2 -d " " > test.txt
     dim_lengthx=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthx") {print $(I+1)};}'`
     dim_lengthy=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthy") {print $(I+1)};}'`
     dim_lengthz=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthz") {print $(I+1)};}'`
     bz=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--bz") {print $(I+1)};}'`
     echo "tensor_size=${dim_lengthx}x${dim_lengthy}x${dim_lengthz}; bz=${bz}" > head.txt
     i=2
  else
     tail -n +2 $f | cut -f 2 -d " " > test2.txt
     paste -d"," test.txt test2.txt > test3.txt
     dim_lengthx=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthx") {print $(I+1)};}'`
     dim_lengthy=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthy") {print $(I+1)};}'`
     dim_lengthz=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthz") {print $(I+1)};}'`
     bz=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--bz") {print $(I+1)};}'`
     echo "tensor_size=${dim_lengthx}x${dim_lengthy}x${dim_lengthz}; bz=${bz}" > head2.txt
     paste -d"," head.txt head2.txt > head3.txt
     cp test3.txt test.txt
     cp head3.txt head.txt
 fi
done
rm test2.txt test3.txt head2.txt head3.txt
cat head.txt > training_concat.csv
cat test.txt >> training_concat.csv
rm head.txt

i=1
for f in *inference*.dat
do

  if [ $i = 1 ]; then
     tail -n +2 $f | cut -f 2 -d " " > test.txt
     dim_lengthx=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthx") {print $(I+1)};}'`
     dim_lengthy=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthy") {print $(I+1)};}'`
     dim_lengthz=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthz") {print $(I+1)};}'`
     bz=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--bz") {print $(I+1)};}'`
     echo "tensor_size=${dim_lengthx}x${dim_lengthy}x${dim_lengthz}; bz=${bz}" > head.txt
     i=2
  else
     tail -n +2 $f | cut -f 2 -d " " > test2.txt
     paste -d"," test.txt test2.txt > test3.txt
     dim_lengthx=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthx") {print $(I+1)};}'`
     dim_lengthy=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthy") {print $(I+1)};}'`
     dim_lengthz=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--dim_lengthz") {print $(I+1)};}'`
     bz=`head -n 1 $f | awk '{for (I=1;I<=NF;I++) if ($I == "--bz") {print $(I+1)};}'`
     echo "tensor_size=${dim_lengthx}x${dim_lengthy}x${dim_lengthz}; bz=${bz}" > head2.txt
     paste -d"," head.txt head2.txt > head3.txt
     cp test3.txt test.txt
     cp head3.txt head.txt
 fi
done
rm test2.txt test3.txt head2.txt head3.txt
cat head.txt > inference_concat.csv
cat test.txt >> inference_concat.csv
rm head.txt

