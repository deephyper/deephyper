ps aux | grep amls | awk '{print $2}' | xargs kill -9
