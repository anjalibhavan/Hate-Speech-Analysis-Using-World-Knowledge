from collections import Counter
import csv
from DataReader import DataReader
from pprint import pprint
from chardet import detect

def get_encoding_type(file):
	with open(file, 'rb') as f:
		rawdata = f.read()
	return detect(rawdata)['encoding']

predicted = []
actual = []
polarity = []
zones = []

full_name = 'filtered.csv'
from_codec = get_encoding_type(full_name)
with open(full_name, 'r', encoding=from_codec,errors='replace') as f, open('temp.csv', 'w', encoding='utf-8') as e:
            text = f.read() # for small files, for big use chunks
            e.write(text)


with open('temp.csv','r') as f:
    r = csv.reader(f)
    for i, line in enumerate(r):
        if i > 0 and i < 301:
            zones.append(line[3])

time = 0
j = 0
for line in zones:
    if line!='India Standard Time':
        print(line)
print('zone',time)


with open('final.csv','r') as f:
    freader = csv.reader(f)
    for i, line in enumerate(freader):
        if i==0:
            continue
        temp = line
        predicted.append(int(temp[0]))
        actual.append(int(temp[1]))
        if temp[2]!='neither':
            polarity.append(int(temp[2]))
        else:
            polarity.append(temp[2])

f = 0
a = 0
n = 0
for i in range(len(polarity)):
    if predicted[i]==0 and actual[i]==1:
        f+=1

print(f)