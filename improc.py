import numpy as np
import scipy.signal as sig
import scipy.ndimage.morphology as m
import scipy.ndimage.measurements as measure
import scipy.ndimage.filters as fil
import matplotlib.pylab as pylb
import mpl_toolkits.mplot3d as mpl_3d
import matplotlib.cm as cm
import matplotlib.image as img
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as interpol
import matplotlib.colors as c

violet = np.array([148, 0, 211])
indigo = np.array([75, 0, 130])
blue = np.array([0, 0, 255])
green = np.array([0, 255, 0])
yellow = np.array([255, 255, 0])
orange = np.array([255, 127, 0])
red = np.array([255, 0, 0])
white = np.array([255, 255, 255])
black = np.array([0, 0, 0])
light_blue = np.array([26, 243, 255])
pop_yellow = np.array([254, 248, 16])
pop_green = np.array([130, 254, 191])
pop_orange = np.array([244, 204, 18])
pop_pink = np.array([241, 157, 242])

kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
kernel_box_blur1 = (1/9) * np.ones((3, 3), dtype='uint8')
kernel_emboss = np.array([[-20, -10, 0], [-10, 0, 10], [0, 10, 20]])
kernel_edge_detection1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
kernel_edge_detection2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel_edge_detection3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel_edge_detection4 = np.array([[1, -1, 1], [-1, 0, -1], [1, -1, 1]])   # my_edge_detect
kernel_box_blur2 = (1/45) * np.ones((5, 5), 'uint8')



def split(image):
	if len(image.shape) > 2:
		return image[:, :, 0], image[:, :, 1], image[:, :, 2]
	return image, image, image


def inRange(image, color1, color2):
	r, g, b = split(image)
	r = r.reshape(len(r), len(r[0]), 1)
	g = g.reshape(len(g), len(g[0]), 1)
	b = b.reshape(len(b), len(b[0]), 1)
	r1 = np.where(color1[0] <= r, [1], [0]).astype('uint8')
	g1 = np.where(color1[1] <= g, [1], [0]).astype('uint8')
	b1 = np.where(color1[2] <= b, [1], [0]).astype('uint8')
	r2 = np.where(r <= color2[0], [1], [0]).astype('uint8')
	g2 = np.where(g <= color2[1], [1], [0]).astype('uint8')
	b2 = np.where(b <= color2[2], [1], [0]).astype('uint8')
	r = np.bitwise_and(r1, r2)
	g = np.bitwise_and(g1, g2)
	b = np.bitwise_and(b1, b2)
	image = np.bitwise_and(r, g)
	image = np.bitwise_and(image, b)
	image = np.repeat(image, 3, axis=1) * 255
	image.shape = (len(r), len(r[0]), 3)
	return image


def color_filter(image, color, width=70):
	color = np.array(color)
	color1 = color - width
	color2 = color + width
	image = inRange(image, color1, color2)
	return image


def moment(image):
	x, y, z = measure.center_of_mass(image)
	return int(x), int(y)


def rescale(image, perc):
	perc = int(perc)
	if perc > 100:
		perc //= 100
		image1 = image.repeat(perc, 0).repeat(perc, 1)
	elif perc == 100:
		image1 = image
	else:
		perc = 100 // perc
		image1 = list()
		for i in range(len(image)):
			if i % perc == 0:
				image1.append(np.array([image[i, j] for j in range(len(image[0])) if j % perc == 0]))
	return np.array(image1)


def flip(image, axis):
	if axis == 'x':
		return image[::-1]
	elif axis == 'y':
		for i in range(len(image)):
			image[i] = image[i][::-1]
		return image


def circle(image, x, y, radius, color=black, thickness=1):
	temp = (np.ones([len(image), len(image[0]), 3]) * 255).astype("uint8")
	print(image.shape, temp.shape)
	x, y, radius = int(x), int(y), int(radius)
	x_c_1 = list(range(x, x + radius + 1))
	y_c_1 = list()
	y_c_2 = list()
	x_c_2 = list()
	for i in range(len(x_c_1)):
		y_c_1.append(int(((radius ** 2 - (x_c_1[i] - x) ** 2) ** 0.5) + y))
		y_c_2.append(int((((radius ** 2 - (x_c_1[i] - x) ** 2) ** 0.5) * -1) + y))
		x_c_2.append(int((((radius ** 2 - (y_c_1[i] - y) ** 2) ** 0.5) * -1) + x))
	x_c = x_c_1 + x_c_1 + x_c_2 + x_c_2
	y_c = y_c_1 + y_c_2 + y_c_1 + y_c_2
	for i in range(len(y_c)):
		temp[(y_c[i])%len(image), (x_c[i])%len(image[0])] = color
	temp = dilation(temp, iterations=thickness-1)
	temp = closing(temp, iterations=3)
	print(image.dtype, temp.dtype)
	image = np.bitwise_and(image, temp)
	return image


def line(image, x1, y1, x2, y2, color=black):
	x = list(range(x1, x2 + 1))
	y = []
	try:
		slope = ((y2 - y1) / (x2 - x1))
		for i in x:
			y.append(int((slope * (i - x1)) + y1))
	except ZeroDivisionError:
		y = list(range(y1, y2))
		x = [x[0]] * len(y)
	for i in range(len(x)):
		image[y[i], x[i]] = color
	return image


def grayscale(image, channels=256):                # new function gives a more precise image
	if 1 < channels < 257:
		high = channels - 1
		image = np.array(image, dtype='int32')
		image = (((image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) * high) / (3 * 255)).round()
		image = (image * 255) / high
		image = np.repeat(image, 3, axis=1)
		image.shape = (len(image), len(image[0]) // 3, 3)
		return np.array(image, dtype='uint8')
	else:
		print("Channels must be within the range of 2 to 256")
		exit(0)


def sharpen(image, iterations=1):
	for i in range(iterations):
		kernel = kernel_sharpen
		r, g, b = split(image)
		r = np.array(sig.convolve2d(r, kernel))
		g = np.array(sig.convolve2d(g, kernel))
		b = np.array(sig.convolve2d(b, kernel))
		r = r.reshape(len(r), len(r[0]), 1)
		g = g.reshape(len(g), len(g[0]), 1)
		b = b.reshape(len(b), len(b[0]), 1)
		image = np.concatenate((r, g, b), axis=2)
	image = normalize(image)
	image = brightness(image, 5)
	image = contrast(image, 5)
	return image


def rotate(image, angle):
	image = interpol.rotate(image, angle=angle)
	return image


def blur(image, iterations=1, threshold=5000 * 3000, rgb=1):
	if len(image) * len(image[0]) > threshold:
		print("Memory error: Image to large. Try a smaller image.")
		exit(0)
	kernel = kernel_box_blur1
	if rgb:
		r = image[:, :, 0]
		g = image[:, :, 1]
		b = image[:, :, 2]
		for i in range(iterations):
			r = sig.convolve2d(r, kernel)#[1:len(image), 1:len(image[0])]
			g = sig.convolve2d(g, kernel)#[1:len(image), 1:len(image[0])]
			b = sig.convolve2d(b, kernel)#[1:len(image), 1:len(image[0])]
		r = r.reshape(len(r), len(r[0]), 1)
		g = g.reshape(len(g), len(g[0]), 1)
		b = b.reshape(len(b), len(b[0]), 1)
		image = np.concatenate((r, g, b), axis=2)
	else:
		image = image[:, :, 0]
		for i in range(iterations):
			image = sig.convolve2d(image, kernel)
		image = image.reshape(len(image), len(image[0]), 1)
		image = np.repeat(image, 3, axis=2)
	return np.array(image, dtype='uint8')


def edge_detect(image, edge_color='w', threshold=5000 * 3000):
	rescaled = 0
	if len(image) * len(image[0]) > threshold:
		image = rescale(image, 25)
		rescaled = 1
	print("rescale check")
	#image = contrast(image, 20)
	#print("contrast")
	image = grayscale(image)
	plt.imshow(image)
	plt.show()
	print("grayscale")
	kernel = kernel_edge_detection3
	image = np.array(sig.convolve2d(image[:, :, 0], kernel)).reshape(len(image) + 2, len(image[0]) + 2, 1)
	print("convolution")
	image = np.repeat(image, 3, 2)
	image = grayscale(normalize(image), 2)
	print("repeat and bw")

	#image = dilation(image, iterations=1)
	#image = erosion(image)
	#image = closing(image, iterations=5)
	#print("erosion and closing")

	if edge_color == 'w':
		if np.count_nonzero(image) > (len(image) * len(image[0]) * 3) / 2:
			image = negetive(image)
	if edge_color == 'b':
		if np.count_nonzero(image) < (len(image) * len(image[0]) * 3) / 2:
			image = negetive(image)
	if rescaled:
		return rescale(image, 400)
	return image


def study_graph(image, color='r'):
	d = {'r': 0, 'g': 1, 'b': 2}
	color = d[color]
	x = np.linspace(0, len(image[0]), len(image[0]), dtype='int32')
	y = np.linspace(0, len(image), len(image), dtype='int32')
	if len(list(np.shape(image))) == 3:
		z = image[:, :, color]
	else:
		z = image[:, :]
	x, y = np.meshgrid(x, y)
	fig = pylb.figure()
	ax = mpl_3d.Axes3D(fig)
	ax.plot_surface(x, y, z, cmap=cm.jet)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	return ax


def normalize(image, mini=-1, maxi=-1):
	if maxi == mini == -1:
		mini = image.min()
		maxi = image.max()
	obj = c.Normalize(mini, maxi)
	norm = obj.__call__(image) * 255
	norm = np.array(norm, dtype='uint8')
	return norm


def negetive(image):
	image = 255 - image
	return image


def equalize(image11, image22):
	image1 = image11.copy()
	image2 = image22.copy()
	x1 = len(image1[0])
	x2 = len(image2[0])
	y1 = len(image1)
	y2 = len(image2)
	x = min([x1, x2])
	y = min([y1, y2])
	if x1 > x:
		image1 = image1[:, :x]
	elif x2 > x:
		image2 = image2[:, :x]
	if y1 > y:
		image1 = image1[:y, :]
	elif y2 > y:
		print(y2 - y)
		image2 = image2[:y, :]
	return image1, image2


def comic(image):
	image1 = grayscale(image, 2)
	image = np.bitwise_and(image, image1)
	return image


def effect2(image):
	image1 = grayscale(image, 2)
	image = np.bitwise_or(image, image1)
	return image


def painting(image):                 # painting
	image1 = grayscale(image, 2)
	image = np.array((image1 / 255) * image, dtype='uint8')
	image = brightness(image, 3)
	image = contrast(image, 2)
	return image


def collage(image):
	image = rescale(image, 50)
	image = grayscale(image, 2) // 255
	image1 = image2 = image3 = image4 = image
	image1 = image1 * pop_yellow
	image2 = image2 * pop_pink
	image3 = image3 * pop_green
	image4 = image4 * orange
	image1 = np.concatenate((image1, image2), axis=1)
	image2 = np.concatenate((image3, image4), axis=1)
	image = np.concatenate((image1, image2), axis=0)
	image = np.array(image, dtype='uint8')
	return image


def brightness(image, level=0, threshold=5000 * 3000):
	if len(image) * len(image[0]) > threshold:
		print("Image to large. Try a smaller image.")
		exit(0)
	if -100 <= level <= 100:
		image = np.array(image, dtype='int32') + level * 10
		if 0 <= level <= 100:
			image = normalize(image, 0, image.max())
		else:
			image = normalize(image, image.min(), 255)
		return np.array(image, dtype='uint8')
	else:
		print("Invalid level. Must be between -100 and 100")


def color_scale(image, color):
	color = np.array(color, dtype='uint8')
	image = (grayscale(image) / 255) * color
	return np.array(image, dtype='uint8')


def emboss(image, iterations=1):
	for i in range(iterations):
		kernel = kernel_emboss
		r = np.array(sig.convolve2d(image[:, :, 0], kernel)).reshape(len(image) + 2, len(image[0]) + 2, 1)
		g = np.array(sig.convolve2d(image[:, :, 1], kernel)).reshape(len(image) + 2, len(image[0]) + 2, 1)
		b = np.array(sig.convolve2d(image[:, :, 2], kernel)).reshape(len(image) + 2, len(image[0]) + 2, 1)
		image = np.concatenate((r, g, b), axis=2)
		image = normalize(image)
	return image


def imread(str):
	return (img.imread(str + '.png') * 255).astype('uint8')



def dilation(image, iterations=1):
	if iterations <= 0:
		return image
	flag = 0
	image = (image / 255).astype("uint8")
	if image.sum() > ((len(image) * len(image[0])) / 2):
		image = 1 - image
		flag = 1
	print(image.sum())
	image1 = m.binary_dilation(image, iterations=iterations)
	image1 = np.where(image1 == True, 255, 0)
	if flag == 1:
		image1 = 255 - image1
	image1 = image1.astype("uint8")
	return image1


def erosion(image, iterations=1):
	if iterations <= 0:
		return image
	flag = 0
	image = (image / 255).astype("uint8")
	if image.sum() > ((len(image) * len(image[0])) / 2):
		image = 1 - image
		flag = 1
	image1 = m.binary_erosion(image, iterations=iterations)
	image1 = np.where(image1 == True, 255, 0)
	if flag == 1:
		image1 = 255 - image1
	image1 = image1.astype("uint8")
	image1 = image1[:, :, 1]
	image1.shape = [len(image1), len(image1[0]), 1]
	image1 = np.repeat(image1, 3, 2)
	return image1


def opening(image, iterations=1):
	if iterations <= 0:
		return image
	flag = 0
	image = (image / 255).astype("uint8")
	if image.sum() > ((len(image) * len(image[0])) / 2):
		image = 1 - image
		flag = 1
	image1 = m.binary_opening(image[:, :, 0], iterations=iterations)
	image1 = np.where(image1 == True, 255, 0)
	image1 = image1.astype("uint8")
	image1.shape = [len(image), len(image[0]), 1]
	image1 = np.repeat(image1, 3, 2)
	if flag == 1:
		image1 = 255 - image1
	image1 = image1.astype("uint8")
	return image1


def closing(image, iterations=1):
	if iterations <= 0:
		return image
	flag = 0
	image = (image / 255).astype("uint8")
	if image.sum() > ((len(image) * len(image[0])) / 2):
		image = 1 - image
		flag = 1
	image1 = m.binary_closing(image[:, :, 0], iterations=iterations)
	image1 = np.where(image1 == True, 255, 0)
	image1 = image1.astype("uint8")
	image1.shape = [len(image), len(image[0]), 1]
	image1 = np.repeat(image1, 3, 2)
	if flag == 1:
		image1 = 255 - image1
	image1 = image1.astype("uint8")
	return image1


def contrast(image, level=1):
	image = image / 255
	image = image - 0.5
	image = 255 / (1 + pylb.exp(-5 * level * image))
	image = image.astype("uint8")
	return image
