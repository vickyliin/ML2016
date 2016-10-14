echo 1 > save.txt
echo kaggle_best.w >> save.txt
python2.7 read_data.py -train -test
python2.7 train.py -feat kaggle_best.f -NOval < save.txt
python2.7 test.py -file kaggle_best.w < save.txt
