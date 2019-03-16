#!/bin/bash

if [[ $1 == 1 ]]; then
	if [[ $4 == "a" ]]; then
		python naive1.py $2 $3 
	elif [[ $4 == "b" ]]; then
		python naive1.py $2 $3 
	elif [[ $4 == "c" ]]; then
		python naive1.py $2 $3 
	elif [[ $4 == "d" ]]; then
		python naive2.py $2 $3 
	elif [[ $4 == "e" ]]; then
		python naivengram.py $2 $3 
	elif [[ $4 == "g" ]]; then
		python naivengram.py $2 $3 	
	fi
elif [[ $1 == 2 ]]; then
	if [[ $4 == 0 ]]; then
		if [[ $5 == "a" ]]; then
			python svmbinary1.py $2 $3 
		elif [[ $5 == "b" ]]; then
			python svmbinary2.py $2 $3 
		elif [[ $5 == "c" ]]; then
			python svmbinary3.py $2 $3 
		fi
	elif [[ $4 == 1 ]]; then
		if [[ $5 == "a" ]]; then
			python svmmulticlass1.py $2 $3 
		elif [[ $5 == "b" ]]; then
			python svmmulticlass2.py $2 $3 
		elif [[ $5 == "c" ]]; then
			python svmmulticlass1.py $2 $3 
		elif [[ $5 == "d" ]]; then
			python svmmulticlass3.py $2 $3 
		fi
	fi
fi
