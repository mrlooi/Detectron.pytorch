from pycocotools.coco import COCO
import numpy as np
import cv2


def view_ann_mask(ann, img):
	m = coco.annToMask(ann)
	height, width, cn_ = img.shape

	mask = np.zeros((height, width), dtype=np.float32)
	# m = m.astype(np.float32) * cat_id
	mask[m>0] = m[m>0]

	# view mask
	mask_view = np.zeros((height, width, cn_), dtype=np.uint8)
	mask_view[m>0] = [255]*cn_ # set all pos to white color
	cv2.imshow('mask', mask_view)
	cv2.waitKey(0)


dataDir ='../data/coco'
dataType ='val2014'
annFile = '{}/annotations/instances_debug2014.json'.format(dataDir)
# annFile = './instances_debug2014.json'

coco=COCO(annFile)

raw_data = coco.dataset

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print 'COCO categories: \n\n', ' '.join(nms)

nms = set([cat['supercategory'] for cat in cats])
print 'COCO supercategories: \n', ' '.join(nms)

catIds = coco.getCatIds(catNms=['person'])#,'dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds )
random_im_id = imgIds[np.random.randint(0,len(imgIds))]

# new_data = {'info': raw_data['info'], 'images': [], 'annotations': [], 
# 		'categories': raw_data['categories'], 'licenses': raw_data['licenses']}
# sample_im_ids = [1000,10012]
sample_im_ids = imgIds
for im_id in sample_im_ids:
	img_data = coco.loadImgs(im_id)[0]		
	annIds = coco.getAnnIds(imgIds=img_data['id'], catIds=catIds, iscrowd=None)
	anns = coco.loadAnns(annIds)

	# VISUALIZE
	img_file = img_data['file_name']
	img_file = '%s/%s/%s'%(dataDir, dataType, img_file)
	img = cv2.imread(img_file)

	cv2.imshow("img", img)
	cv2.waitKey(0)

	for ann in anns:
		view_ann_mask(ann, img)  #  binary mask

	# new_data['images'].append(img_data)
	# new_data['annotations'] += anns

# import json
# with open("instances_debug2014.json", "w") as f:
# 	json.dump(new_data, f)
