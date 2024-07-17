import os

images = []
for file in os.listdir("/root/autodl-tmp/52000"):
    if file.endswith(".png"):
        images.append(file)

with open("data/fhq.txt", "w") as f:
    for image in images:
        f.write("52000/" + image + "\n")
