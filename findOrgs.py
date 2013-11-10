#!/usr/bin/env python

train = open('./data/dutch.train.txt', 'r')

target = 'ORG'

orgs = []
for line in train.readlines():
  if (len(line) > 1):
    tag = line.split()[2]
    if (tag[-len(target):] == target):
      orgs += [line]

orgs = sorted(orgs)
orgsCounts = []

cnt = 1
for i, org in enumerate(orgs):
  if (i > 0):
    if (orgs[i - 1] == orgs[i]):
      cnt = cnt + 1
    else:
      orgsCounts += [(cnt, orgs[i - 1])]
      cnt = 1

if cnt:
  orgsCounts += [(cnt, orgs[-1])]

orgsCounts = sorted(orgsCounts, reverse=True)
for org in orgsCounts:
  print(org[1], ' ', org[0])
