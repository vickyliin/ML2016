echo > .enter
python test.py -dir $1 -load $2 -save $3 < .enter
rm .enter
