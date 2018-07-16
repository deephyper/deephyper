#!/bin/bas
rm $BALSAM_DB_PATH/log/*
rm -rf $BALSAM_DB_PATH/data/*
balsam rm jobs --name task
balsam modify jobs 64f --attr state --value CREATED
balsam launcher --c --serial-jobs=4 --job-mode serial
