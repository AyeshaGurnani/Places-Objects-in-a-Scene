import final_scene_recogition as scene_rec
import object_recognition as obj_rec
import sys, os
import json
import random
import cv2
import operator

scenes_file = 'scenes.txt'
# dataset_file = '/Users/ketan/Downloads/result.json'
dataset_file = 'result.json'
object_dataset = 'cropped_objects/'
output_dir = 'output_dir/'

# image = "../images/Places365_test_00001489.jpg"
# image = "../images/Places365_test_00001494.jpg"
# image = "test2.jpeg"

if len(sys.argv) > 1:
    image = sys.argv[1]
else:
    "please pass an image as an argument"
    sys.exit()
try:
    image_array = cv2.imread(image)
except:
    print "Please pass the image with correct path"
    sys.exit()

scene_type = scene_rec.run(image)
existing_objs = obj_rec.run(image)

with open(dataset_file, 'r') as f:
    data = json.load(f)

# print data
with open(scenes_file, 'r') as f:
    scenes = f.read().split(',')
    # print scenes
print scenes
our_background = None


for scene in scene_type:
    # scene = unicode(scene, "utf-8")
    print scene
    if scene in scenes:
        our_background = scene
        break

print our_background
if not our_background:
    print "Scene is not in our dataset! Please try with another image."
    sys.exit()
object = data[our_background]

dict = {}
for keys in object:
    dict[keys[0]] = keys[1]

for key, value in dict.items():
    if key == 'person':
        del dict[key]
print dict


objects = sorted(dict.items(), key=operator.itemgetter(1), reverse = True)
print objects



print existing_objs

objs = [ item['label'] for item in existing_objs ]

objects_to_place = 2

dont_place_here = []

remained_objects = []

for object in objects:
    flag = False
    for i in range(len(objs)):
        if unicode(objs[i], 'UTF-8') == object[0]:
            dont_place_here.append([[existing_objs[i]['topleft']['x'], existing_objs[i]['topleft']['y']], [existing_objs[i]['bottomright']['x'], existing_objs[i]['bottomright']['y']]])
            flag = True
            continue
    if not flag:
        remained_objects.append(object)


size = image_array.shape
xt = size[1]/5
yt = size[0]/5

placed_objs = []

def place_obj(object, max_n = 20):
    x = random.randint(0, size[1])
    y = random.randint(0, size[0])
    for i in dont_place_here:
        if x in range(i[0][0], i[1][0]) or x + xt  in range(i[0][0], i[1][0]) or y in range(i[0][1], i[1][1]) or y + yt  in range(i[0][1], i[1][1]) or x + xt >= size[1] or y + yt >= size[0]:
            if max_n == 0:
                return False
            return place_obj(object, max_n=max_n - 1)
        else:
            continue
    our_object_name = object_dataset + object + '/'
    for _, _, samples in os.walk(our_object_name):
        a = random.randint(0, len(samples) - 1)
        our_object = our_object_name + samples[a]
        our_object_ = cv2.resize(cv2.imread(our_object), dsize=(xt, yt), interpolation=cv2.INTER_CUBIC)
        try:
            image_array[y:y + yt, x:x + xt] = our_object_
        except:
            return place_obj(object, max_n=max_n - 1)
        placed_objs.append(object)
        dont_place_here.append([[x, xt], [y, yt]])
        break
    return True


for object in remained_objects:

    status = place_obj(object[0].encode('UTF-8'))

    if status:
        objects_to_place -= 1
    if objects_to_place == 0:
        break

output_nu = str(random.randint(0, 100))
cv2.imwrite(output_dir + output_nu + '.jpg', image_array)

print "We have placed ", placed_objs, " objects!"
print "Your output image with object places is at " + output_dir + output_nu + ".jpg!"

