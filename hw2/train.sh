echo > .enter
echo >> .enter
python train.py -in $1 -out $2 -noScale -para:eta,itr 0.025,10000 -noVal < .enter
rm .enter
