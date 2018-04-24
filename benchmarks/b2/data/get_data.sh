#!/usr/bin/bash

wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
if [ $? -ne 0 ]
then
    curl http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz -o tasks_1-20_v1-2.tar.gz
fi

mv tasks_1-20_v1-2.tar.gz babi-tasks-v1-2.tar.gz
