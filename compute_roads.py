import argparse
import os
from IPython import embed
from util import util
import models.dist_model as dm
import sys

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dir', type=str, default='./imgs/ex_dir0')
	# parser.add_argument('--out', type=str, default='./imgs/example_dists.txt')
	parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
	opt = parser.parse_args()

	## Initializing the model
	model = dm.DistModel()
	model.initialize(model='net-lin',net='alex',use_gpu=opt.use_gpu)

	res = 0.0
	files = os.listdir(opt.dir)
	files.sort()
	ext = ['.png','.jpg','.jpeg',]
	files = [file for file in files if os.path.splitext(file)[1] in ext ]
	# print(files)
	outpath = os.path.join(opt.dir,'distance_{}.txt'.format(len(files)))

	# crawl directories
	with open(outpath,'w',encoding='utf-8') as f:
		for i,file in enumerate(files):
			if i < len(files)-1 and os.path.exists(os.path.join(opt.dir,files[i])) and os.path.exists(os.path.join(opt.dir,files[i+1])):
				# Load images
				img0 = util.im2tensor(util.load_image(os.path.join(opt.dir,file))) # RGB image from [-1,1]
				img1 = util.im2tensor(util.load_image(os.path.join(opt.dir,files[i+1])))

				# Compute distance
				dist01 = model.forward(img0,img1)
				res += dist01
				f.writelines('{} {} {}'.format(file,files[i+1],dist01))

				if i % 10 == 0:
					sys.stdout.write('\r')
					sys.stdout.write('Computing distances... {:.2f} %'.format(i/len(files)*100.))
					sys.stdout.flush()

		res /= (len(files)-1)
		f.writelines('\n### result average distance is: {}'.format(res))
		print('\n### result average distance is: ', res)