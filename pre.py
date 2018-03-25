from resizeimage import resizeimage
from PIL import Image
import os

def resize_them(path,outpath,size):
	if not os.path.isdir(path):
		raise Exception('Path doesnt exist')
	if not os.path.isdir(outpath):
		raise Exception('outpath doesnt exist')
			
	for file in os.listdir(path):
		orig_path = os.path.join(path,file)
		img = Image.open(orig_path)
		hratio = size[0]/img.size[0]
		wratio = size[1]/img.size[1]
		xshape = min(hratio,wratio)
		if xshape > 1:
			enlarged_size = (int(img.size[0]*xshape),int(img.size[1]*xshape))
			img = img.resize(enlarged_size)
		img = resizeimage.resize_contain(img,size)
		img.save(os.path.join(outpath,file),img.format)


