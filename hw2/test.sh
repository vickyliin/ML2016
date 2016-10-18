echo > .enter
echo >> .enter
python test.py -model $1 -test $2 -out $3 < .enter
rm .enter
