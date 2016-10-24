echo > .enter
echo >> .enter
python train.py -in $1 -out $2 -noScale -para:eta 0.025 -noVal < .enter
rm .enter
