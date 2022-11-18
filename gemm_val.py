import torch


fi = open("build/input.txt", 'r')
lines = fi.readlines()
row = len(lines)
# row = 1

input = []
weight = []
output = []

for line in lines[0:row]:
    line = line.strip().split()
    temp = [int(i) for i in line]
    input.append(temp)

print("-----1----")
print(input)

fi = open("build/weight_plain.txt", 'r')
lines = fi.readlines()

for line in lines:
    line = line.strip().split()
    temp = [int(i) for i in line]
    weight.append(temp)

print("-----2----")
print(weight)

fi = open("build/output.txt", 'r')
lines = fi.readlines()

for line in lines[0:row]:
    line = line.strip().split()
    temp = [int(i) for i in line]
    output.append(temp)

print("-----3----")

it = torch.tensor(input)
wt = torch.tensor(weight)
ot = torch.tensor(output)

print(it)
print(wt)
print(ot)

print("-----4----")

res = torch.matmul(it, wt)

print("-----5----")
print(ot)
print(res)

print(torch.sum(ot))
print(torch.sum(res))


