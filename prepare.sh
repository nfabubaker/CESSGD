#!/bin/bash

coan source -UDEBUG -UNA_DBG --filter c,h --recurse src 

for i in src/*
do
  if ! grep -q Copyright $i
  then
    cat copyright.c $i >$i.new && mv $i.new $i
  fi
done

