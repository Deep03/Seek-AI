import handTrack as ht
import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

lmlist1 = ht.main()
print(lmlist1)
wrist_arr = lmlist1[0]
thumb1 = lmlist1[2]
thumb2 = lmlist1[4]
index1 = lmlist1[5]
index2 = lmlist1[8]
middle1 = lmlist1[9]
middle2 = lmlist1[12]
ring1 = lmlist1[13]
ring2 = lmlist1[16]
pinky1 = lmlist1[17]
pinky2 = lmlist1[20]
thumb_dist = distance(thumb1[1], thumb1[2], thumb2[1], thumb2[2])
index_dist = distance(index1[1], index1[2], index2[1], index2[2])
middle_dist = distance(middle1[1], middle1[2], middle2[1], middle2[2])
ring_dist = distance(ring1[1], ring1[2], ring2[1], ring2[2])
pinky_dist = distance(pinky1[1], pinky1[2], pinky2[1], pinky2[2])
max_finger = max(thumb_dist, index_dist, middle_dist, ring_dist, pinky_dist)
if (max_finger == thumb_dist):
    with open("/Users/rootapollo/Documents/Docs/Seek-AI/seek/index.txt", 'w+') as f:
        f.write("IT WORKED!!!!\n")
        f.close()
else:
    print("Task Failed Succesfully")
