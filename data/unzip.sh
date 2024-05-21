#!/bin/bash

unzip -n files-archive

for i in 0 1 2 3 4 5 6 7 8 9
do
   unzip -qn "*$i9.zip"
done

mv DREAM*/* .
rm -r DREAM*
