echo > .enter
echo >> .enter
python train.py -in $1 -out $2 -stop 0.8 -noVal < .enter
rm .enter
