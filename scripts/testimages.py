path_start = "../images/empty-boards/empty-"

if im_no == 3:
    image_path = path_start + "3.png"
elif im_no in [6,7]:
    image_path = path_start + str(im_no) + ".jpeg"
else:
    image_path = path_start + str(im_no) + ".jpg"

