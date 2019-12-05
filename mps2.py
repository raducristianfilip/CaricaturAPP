#from PIL import Image
from PIL import Image
import cv2
import argparse
import math
import progressbar
from pointillism import *


def black_and_white(input_image_path,
                    output_image_path):
	color_image = Image.open(input_image_path)
	bw = color_image.convert('L')
	bw.save(output_image_path)


def make_sepia_palette(color):
	palette = []
	r, g, b = color
	for i in range(255):
		palette.extend(((r * i) / 255, (g * i) / 255, (b * i) / 255))

	return palette


def create_sepia(input_image_path, output_image_path):
	whitish = (255, 240, 192)
	sepia = make_sepia_palette(whitish)
	sepia = [round(x) for x in sepia]

	color_image = Image.open(input_image_path)

	# convert our image to gray scale
	bw = color_image.convert('L')

	# add the sepia toning
	bw.putpalette(sepia)

	# convert to RGB for easier saving
	sepia_image = bw.convert('RGB')

	sepia_image.save(output_image_path)

def create_cartoon(input_image_path, output_image_path):
	num_down = 2  # number of downsampling steps
	num_bilateral = 7  # number of bilateral filtering steps

	img_rgb = cv2.imread(input_image_path)

	# downsample image using Gaussian pyramid
	img_color = img_rgb
	for _ in range(num_down):
		img_color = cv2.pyrDown(img_color)

	# repeatedly apply small bilateral filter instead of
	# applying one large filter
	for _ in range(num_bilateral):
		img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

	# upsample image to original size
	for _ in range(num_down):
		img_color = cv2.pyrUp(img_color)

	# convert to grayscale and apply median blur
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
	img_blur = cv2.medianBlur(img_gray, 7)

	# detect and enhance edges
	img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

	# convert back to color, bit-AND with color image
	img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
	img_cartoon = cv2.bitwise_and(img_color, img_edge)

	cv2.imwrite(output_image_path, img_cartoon)

def create_drawing(input_image):
	parser = argparse.ArgumentParser(description='...')
	parser.add_argument('--palette-size', default=20, type=int, help="Number of colors of the base palette")
	parser.add_argument('--stroke-scale', default=0, type=int, help="Scale of the brush strokes (0 = automatic)")
	parser.add_argument('--gradient-smoothing-radius', default=0, type=int,
	                    help="Radius of the smooth filter applied to the gradient (0 = automatic)")
	parser.add_argument('--limit-image-size', default=0, type=int, help="Limit the image size (0 = no limits)")
	parser.add_argument('img_path', nargs='?', default=input_image)

	args = parser.parse_args()

	res_path = args.img_path.rsplit(".", -1)[0] + "_drawing.jpg"
	img = cv2.imread(args.img_path)

	if args.limit_image_size > 0:
		img = limit_size(img, args.limit_image_size)

	if args.stroke_scale == 0:
		stroke_scale = int(math.ceil(max(img.shape) / 1000))
		print("Automatically chosen stroke scale: %d" % stroke_scale)
	else:
		stroke_scale = args.stroke_scale

	if args.gradient_smoothing_radius == 0:
		gradient_smoothing_radius = int(round(max(img.shape) / 50))
		print("Automatically chosen gradient smoothing radius: %d" % gradient_smoothing_radius)
	else:
		gradient_smoothing_radius = args.gradient_smoothing_radius

	# convert the image to grayscale to compute the gradient
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	print("Computing color palette...")
	palette = ColorPalette.from_image(img, args.palette_size)

	print("Extending color palette...")
	palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

	# display the color palette
	#cv2.imshow("palette", palette.to_image())
	cv2.waitKey(200)

	print("Computing gradient...")
	gradient = VectorField.from_gradient(gray)

	print("Smoothing gradient...")
	gradient.smooth(gradient_smoothing_radius)

	print("Drawing image...")
	# create a "cartonized" version of the image to use as a base for the painting
	res = cv2.medianBlur(img, 11)
	# define a randomized grid of locations for the brush strokes
	grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
	batch_size = 10000

	bar = progressbar.ProgressBar()
	for h in bar(range(0, len(grid), batch_size)):
		# get the pixel colors at each point of the grid
		pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])
		# precompute the probabilities for each color in the palette
		# lower values of k means more randomnes
		color_probabilities = compute_color_probabilities(pixels, palette, k=9)

		for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
			color = color_select(color_probabilities[i], palette)
			angle = math.degrees(gradient.direction(y, x)) + 90
			length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))

			# draw the brush stroke
			cv2.ellipse(res, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)

	#cv2.imshow("res", limit_size(res, 1080))
	cv2.imwrite(res_path, res)
	cv2.waitKey(0)

if __name__ == '__main__':
	black_and_white('vali.jpg', 'bw_eu.jpg')
	print("-------------------Black and White Done----------------")
	create_sepia('vali.jpg', 'sepia_eu.jpg')
	print("--------------------Sepia Done-------------------------")
	create_cartoon('vali.jpg', 'cartoon_eu.jpg')
	print("----------------------Cartoon Done-----------------------")
	create_drawing('cartoon_eu.jpg')
	print("------------------------Drawing Done----------------------")