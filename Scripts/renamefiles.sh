#!/bin/(shell)

#  Use this to count everything in subdirectories: 
# ls | xargs -I{} ls {} | wc -l


count=0
# var="words.txt"

while read line; 
do 
for file in FinalTrain/"$line"/*
do 
let count++ 
# echo $file
mv "$file" FinalTrain/"$line"/"$count".png;
# echo $count 
# echo $line
done;
# ls /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTrainingData/"$line2" | head -5 | xargs -I{} mv /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTrainingData/"$line2"/{} /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTestingData/"$line"/;
done < folders.txt 