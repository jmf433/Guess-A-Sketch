#!/bin/(shell)

while read line; 
do 
mkdir FinalTest/"$line" 
done < folders.txt
