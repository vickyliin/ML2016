echo 1 > save.txt
echo linear_regression.w >> save.txt
python2.7 read_data.py -train -test
python2.7 train.py -feat linear_regression.f -reg1 -itr-0.001 -NOval < save.txt
python2.7 test.py -file linear_regression.w < save.txt
