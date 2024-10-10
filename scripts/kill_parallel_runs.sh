#!/bin/bash

# Print the current list of Python processes
echo "Current Python processes:"
ps aux | grep 'python' | grep -v grep

# Search for processes matching "some search" and kill them
for pid in $(ps -ef | grep "harvest" | grep -v grep | awk '{print $2}'); do
    kill -9 $pid
    echo "Killed process with PID: $pid"
done
# Search for processes matching "some search" and kill them
for pid in $(ps -ef | grep "scripts" | grep -v grep | awk '{print $2}'); do
    kill -9 $pid
    echo "Killed process with PID: $pid"
done

echo "Python processes after killing:"
ps aux | grep 'python' | grep -v grep