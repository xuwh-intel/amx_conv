import os

f = open("build/weight_block.txt", 'r')

lines = f.readlines()

ic, oc = 32, 64

weight_blocked = [[] * 64] * 32
weight_plain = [[0 for i in range(32)] for j in range(64)]

for i in range(len(lines)):
    line = lines[i].strip().split()
    weight_blocked[i] = [int(k) for k in line]


for i in range(16):
    temp = []
    for j in range(32):
        for k in range(4):
            temp.append(weight_blocked[j][i * 4 + k])
    print(len(temp))
    for jj in range(64):
        weight_plain[jj][i] = temp[jj]
        weight_plain[jj][i + 16] = temp[jj + 64]



# # bert
# for k in range(4):
#     for i in range(64 // 4):
#         temp = []
#         for j in range(64):
#             for l in range(4):
#                 temp.append(weight_blocked[64 * k + j][(i * 4 + l)])
        
#         for jj in range(256):
#             weight_plain[jj][k * 16 + i] = temp[jj]


# for i in range(256 // 4):
#     temp = []
#     for j in range(64):
#         for k in range(4):
#             temp.append(weight_blocked[j][(i * 4 + k)])
#     # print(temp)
#     # break
#     for jj in range(256):
#         weight_plain[jj][i] = temp[jj]

sf = open("build/weight_plain.txt", 'w')

for i in range(64):
    buf = ""
    for j in range(32):
        buf += str(weight_plain[i][j]) + ' '
    buf += "\n"
    sf.write(buf)
sf.close()



# for i in range(64):
#         print(weight_blocked[i])
#     # print("\n")