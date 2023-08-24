#!/bin/(shell)

temp=0
temp2=1

while read line; 
do
let temp++;
let temp2=1;
while read line2;
do 
if [ "$temp" -eq "$temp2" ]
then 
ls /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTrainingData/"$line2" | head -5 | xargs -I{} mv /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTrainingData/"$line2"/{} /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTestingData/"$line"/;
# echo $temp 
# echo $line
# echo $line2
fi
let temp2++; 
# for i in $(ls /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTrainingData); do 
# ls i | head -5 | xargs -I{} mv {} /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTestingData/$line/;
# done; 
done < words2.txt; done < words.txt  


# for i in $(ls *.pdf); do
# mv $i $(basename $i .pdf)_$(date +%Y%m%d).pdf
# done

# ls -1 | head -5 | xargs -I{} mv {} /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTestingData/ant/


# do for i in $(ls /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTrainingData)



# while read line; while read line2; do 
# ls $line | head -5 | xargs -I{} mv {} /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/NewTestingData/$line2/;
# done < /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/words2.txt; done < /Users/johnfernandez/Documents/"Spring 23"/"CS 4701"/Guess-a-Sketch/words.txt