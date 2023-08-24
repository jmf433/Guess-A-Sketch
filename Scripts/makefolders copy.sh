#!/bin/(shell)

while read line; 
do 
mkdir 256Train/"$line"; 
mkdir 256Test/"$line"
mkdir 256Val/"$line"
done < folders.txt
