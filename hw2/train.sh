echo > .enter
echo >> .enter
python train.py -in $1 -out $2 -feat drop:cf -para:eta 100 -noVal < .enter
rm .enter
