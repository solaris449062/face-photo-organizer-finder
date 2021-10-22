import face_recognition # import face recognition library
import matplotlib.pyplot as plt # import matplotlib plot
import matplotlib.image as mpimg # import matplotlib image
import matplotlib.patches as patches # import matplotlib patch drawing
from PIL import Image # import python image processing library

# cv2 makes the VS Code extension crash and kernel cannot be restarted. 

workdir = "./"
filename = "infinitywar_test.jpg"

img_path = workdir + filename # image path assignment

PIL_image = Image.open(img_path)
print(PIL_image.size) # gives file size

image = face_recognition.load_image_file(img_path)
face_locations = face_recognition.face_locations(image)
# face_locations = face_recognition.face_locations(image, model="cnn")

print(str(len(face_locations)) + " face(s) detected")
print(face_locations)

figure_original, ax_original = plt.subplots() # create figure
face_regions = [0]*len(face_locations) # populate array for the rectangles to be drawn around the recognized faces

for i in range(len(face_regions)): # add red rectangles according to the coordinates
    face_regions[i] = patches.Rectangle((face_locations[i][3],face_locations[i][0]),abs(face_locations[i][2]-face_locations[i][0]),abs(face_locations[i][3]-face_locations[i][1]), edgecolor='r', facecolor="none")

matplotlib_image = mpimg.imread(img_path)
ax_original.set_axis_off() # this removes the axis from the photo

for i in range(len(face_locations)): # adds red rectangle around found faces
    ax_original.add_patch(face_regions[i])

ax_original.imshow(matplotlib_image) # show image with found faces


# export the face-detected photos into files
save_all_filename = filename[slice(-4)] + "_faces_all.jpg"
figure_original.savefig(save_all_filename, dpi=150, bbox_inches="tight")  # entire photo with all faces with rectangles around found faces




# PIL has (left, top, right, bottom) system. This constructs a rectangle from (left, top) coordinate to (right, bottom) coordinate
# face-recognition has (top, right, bottom, left) system. This constructs a rectangle from (top, right) to (bottom, left) coordinate

def coord_transform_face_rec_to_PIL(coord_face_recognition):
    # from (top, right, bottom, left) system to (left, top, right, bottom) system
    (top, right, bottom, left) = coord_face_recognition
    coord_PIL = (left, top, right, bottom)
    return coord_PIL

# print(coord_transform_face_rec_to_PIL((188, 895, 239, 844)))

save_face_filename = filename[slice(-4)] + "_face.jpg"

for i in range(len(face_locations)): # exports the faces-only cropped region into files
    img_cropped = PIL_image.crop(coord_transform_face_rec_to_PIL(face_locations[i]))
    # img_cropped.show()
    img_cropped.save(filename[slice(-4)] + "_face_" + str(i) + ".jpg")