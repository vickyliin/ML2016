echo > .enter
python train.py -dir $1 -save $2 <.enter
rm .enter
