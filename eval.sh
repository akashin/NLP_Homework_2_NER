#!/usr/bin/sh

if [ -f dutch.data ]; then
  rm dutch.data
fi

if [ -f dutch.maxent ]; then
  rm dutch.maxent
fi

echo "Generate model"
./run.sh -f data/dutch.train.txt.train "$@" -m dutch -T

echo "Generate prediction"
./run.sh -f data/dutch.train.txt.test -m dutch > data/result-dutch

echo "Evaluate score"
(cd data && ./eval.sh dutch.train.txt.test.ans result-dutch)
