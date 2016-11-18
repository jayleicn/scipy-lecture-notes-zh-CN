.. for doctests
   >>> import numpy as np
   >>> np.random.seed(0)
   >>> import matplotlib.pyplot as plt
   >>> plt.switch_backend("Agg")


.. _basic_image:

=======================================================
使用 Numpy 和 Scipy 进行图片处理
=======================================================

**作者**: *Emmanuelle Gouillart, Gaël Varoquaux*

这一节旨在使用Numpy和SciPy来解决基础图像处理问题。同时，本节中所提到的某些操作对一些形式的多维矩阵也有帮助。比如，:mod:`scipy.ndimage` 提供了一些对NumPy矩阵进行操作的函数。

.. seealso::

    对于一些更高级的图像处理问题和教程，请参考 :ref:`scikit_image`。

.. topic::
    图像 = 2-D 数值矩阵

    (或者 3-D: CT, MRI, 2D + 时间; 4-D, ...)

    这里, **图像 == Numpy矩阵** ``np.array``

**本节所用工具**:

* ``numpy``: 基础矩阵操作

* ``scipy``: ``scipy.ndimage`` ，一个处理n维图像的子包。`参考文档
  <http://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html>`_::

    >>> from scipy import ndimage


**图像处理的常见任务**:

* 图像的输入输出及展示

* 基础处理程序: 裁剪, 翻转, 旋转, ...

* 图像滤波: 去燥, 提高清晰度

* 图像分割: 对属于不同物体的像素贴上不同的标签

* 分类

* 特征提取

* ...


.. contents:: 目录
   :local:
   :depth: 4



打开和写入图像文件
==================================

将矩阵写入文件:

.. literalinclude:: examples/plot_face.py
   :lines: 8-

.. image:: examples/face.png
    :align: center
    :scale: 50

从numpy矩阵创建图像::

    >>> from scipy import misc
    >>> face = misc.face()
    >>> misc.imsave('face.png', face) # 首先我们需要创建一个PNG文件
    
    >>> face = misc.imread('face.png')
    >>> type(face)      # doctest: +ELLIPSIS
    <... 'numpy.ndarray'>
    >>> face.shape, face.dtype
    ((768, 1024, 3), dtype('uint8'))

数据类型dtype是“uint8”，表示每个像素用8位二进制存储(0-255)

打开文件::

    >>> face.tofile('face.raw') # 创建文件
    >>> face_from_raw = np.fromfile('face.raw', dtype=np.uint8)
    >>> face_from_raw.shape
    (2359296,)
    >>> face_from_raw.shape = (768, 1024, 3)

此处需要已知图片的尺寸。

大文件使用``np.memmap``进行内存映射::

    >>> face_memmap = np.memmap('face.raw', dtype=np.uint8, shape=(768, 1024, 3))

(数据从文件读入，但不载入内存)

操作一系列图片 ::

    >>> for i in range(10):
    ...     im = np.random.random_integers(0, 255, 10000).reshape((100, 100))
    ...     misc.imsave('random_%02d.png' % i, im)
    >>> from glob import glob
    >>> filelist = glob('random*.png')
    >>> filelist.sort()

显示图片
=================

使用``matplotlib``和``imshow``在``matplotlib figure``中显示图片::

    >>> f = misc.face(gray=True)  # 创建一张灰度图
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(f, cmap=plt.cm.gray)        # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at 0x...>

在上图基础上通过设置min和max来增加对比度::

    >>> plt.imshow(f, cmap=plt.cm.gray, vmin=30, vmax=200)        # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at 0x...>
    >>> # 去掉坐标轴和刻度
    >>> plt.axis('off')
    (-0.5, 1023.5, 767.5, -0.5)

添加轮廓线::

    >>> plt.contour(f, [50, 200])        # doctest: +ELLIPSIS
    <matplotlib.contour.QuadContourSet ...>


.. figure:: auto_examples/images/plot_display_face_1.png
    :scale: 80
    :target: auto_examples/plot_display_face.html

.. only:: html

    [:ref:`Python source code <example_plot_display_face.py>`]

使用``interpolation='nearest'``来增加分辨率::

    >>> plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray)        # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at 0x...>
    >>> plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray, interpolation='nearest')        # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at 0x...>

.. figure:: auto_examples/images/plot_interpolation_face_1.png
    :scale: 80
    :target: auto_examples/plot_interpolation_face.html

.. only:: html

    [:ref:`Python source code <example_plot_interpolation_face.py>`]


.. seealso:: 3-D 可视化工具: Mayavi

    文档 :ref:`mayavi-label`.

	* Image plane widgets

	* Isosurfaces

	* ...

.. image:: ../../packages/3d_plotting/decorations.png
    :align: center
    :scale: 65


基础操作
===================

图像是多维矩阵: 可以使用``numpy``进行很多操作

.. image:: axis_convention.png
    :align: center
    :scale: 65

::

    >>> face = misc.face(gray=True)
    >>> face[0, 40]
    127
    >>> # 截取
    >>> face[10:13, 20:23]
    array([[141, 153, 145],
           [133, 134, 125],
           [ 96,  92,  94]], dtype=uint8)
    >>> face[100:120] = 255
    >>>
    >>> lx, ly = face.shape
    >>> X, Y = np.ogrid[0:lx, 0:ly]
    >>> mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
    >>> # 选定一些特殊位置
    >>> face[mask] = 0
    >>> # 高级索引
    >>> face[range(400), range(400)] = 255

.. figure:: auto_examples/images/plot_numpy_array_1.png
    :scale: 100
    :target: auto_examples/plot_numpy_array.html

.. only:: html

    [:ref:`Python source code <example_plot_numpy_array.py>`]


统计信息
-----------------------

::

    >>> face = misc.face(gray=True)
    >>> face.mean()
    113.48026784261067
    >>> face.max(), face.min()
    (250, 0)


``np.histogram``

.. topic:: **练习**
    :class: green
    

    * 读入一张图片，比如``scikit-image``图标(http://scikit-image.org/_static/img/logo.png)。

    * 截取图片的一部分，比如上述图标中的蟒蛇圆圈。

    * 用``matplotlib``显示图像矩阵. 改变差值方法，放大看有什么不同。

    * 将你的图片转换为灰度图。

    * 通过改变最值来增加图片的对比度。 **选做**: 使用``scipy.stats.scoreatpercentile``
      将最暗和最亮的5%像素达到饱和。

    * 将图片存储为不同的格式 (png, jpg, tiff)

    .. image:: scikit_image_logo.png
        :align: center


图像的几何变换
---------------------------
::

    >>> face = misc.face(gray=True)
    >>> lx, ly = face.shape
    >>> # 截取
    >>> crop_face = face[lx / 4: - lx / 4, ly / 4: - ly / 4]
    >>> # 上下翻转
    >>> flip_ud_face = np.flipud(face)
    >>> # 旋转
    >>> rotate_face = ndimage.rotate(face, 45)
    >>> rotate_face_noreshape = ndimage.rotate(face, 45, reshape=False)

.. figure:: auto_examples/images/plot_geom_face_1.png
    :scale: 65
    :target: auto_examples/plot_geom_face.html

.. only:: html

    [:ref:`Python source code <example_plot_geom_face.py>`]

图片滤波
===============

**局部滤波**: 将像素的值用其周围像素值的一个函数替代。

Neighbourhood: square (choose size), disk, or more complicated *structuring
element*.

.. figure:: kernels.png
    :align: center
    :scale: 90

模糊／平滑
------------------

``scipy.ndimage``中的**高斯滤波**::

    >>> from scipy import misc
    >>> face = misc.face(gray=True)
    >>> blurred_face = ndimage.gaussian_filter(face, sigma=3)
    >>> very_blurred = ndimage.gaussian_filter(face, sigma=5)

**均值滤波** ::

    >>> local_mean = ndimage.uniform_filter(face, size=11)

.. figure:: auto_examples/images/plot_blur_1.png
    :scale: 90
    :target: auto_examples/plot_blur.html

.. only:: html

    [:ref:`Python source code <example_plot_blur.py>`]

锐化
----------

锐化模糊图像::

    >>> from scipy import misc
    >>> face = misc.face(gray=True).astype(float)
    >>> blurred_f = ndimage.gaussian_filter(face, 3)

通过加上拉普拉斯近似来增加边缘权重::

    >>> filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    >>> alpha = 30
    >>> sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

.. figure:: auto_examples/images/plot_sharpen_1.png
    :scale: 65
    :target: auto_examples/plot_sharpen.html

.. only:: html

    [:ref:`Python source code <example_plot_sharpen.py>`]


去燥
---------

有噪点的图片::

    >>> from scipy import misc
    >>> f = misc.face(gray=True)
    >>> f = f[230:290, 220:320]
    >>> noisy = f + 0.4 * f.std() * np.random.random(f.shape)

使用**高斯滤波**抹掉噪点，但这样也会使边缘变得模糊::

    >>> gauss_denoised = ndimage.gaussian_filter(noisy, 2)

大部分局部线性各向同性过滤器都会使图像变得模糊(``ndimage.uniform_filter``)

而**均值滤波器**可以较好的保存边缘信息::

    >>> med_denoised = ndimage.median_filter(noisy, 3)

.. figure:: auto_examples/images/plot_face_denoise_1.png
    :scale: 60
    :target: auto_examples/plot_face_denoise.html

.. only:: html

    [:ref:`Python source code <example_plot_face_denoise.py>`]


均值滤波器: 对棱角分明的图片更好 (**曲度小**)::

    >>> im = np.zeros((20, 20))
    >>> im[5:-5, 5:-5] = 1
    >>> im = ndimage.distance_transform_bf(im)
    >>> im_noise = im + 0.2 * np.random.randn(*im.shape)
    >>> im_med = ndimage.median_filter(im_noise, 3)

.. figure:: auto_examples/images/plot_denoising_1.png
    :scale: 50
    :target: auto_examples/plot_denoising.html

.. only:: html

    [:ref:`Python source code <example_plot_denoising.py>`]


其它各阶滤波器: ``ndimage.maximum_filter``,
``ndimage.percentile_filter``

其它局部非线性滤波器: Wiener (``scipy.signal.wiener``), 等等.

**非局部滤波器**

.. topic:: **练习：去燥**
    :class: green
    
    * 创建一张包含圆，椭圆，正方形或者任意形状物体的二值图片(只含0，1)。

    * 添加噪点

    * 尝试两种不同的去燥方法: 高斯滤波和均值滤波.

    * 使用直方图比较两者的不同。哪一个和不含噪点的图片的直方图更相近？

.. seealso::

    更多去燥的滤波器 :mod:`skimage.denoising`,
    也可以看看 :ref:`scikit_image` 的教程.



数学形态学
-----------------------

参见 `wikipedia <https://en.wikipedia.org/wiki/Mathematical_morphology>`_
了解数学形态学的定义。


**结构化元素**::

    >>> el = ndimage.generate_binary_structure(2, 1)
    >>> el
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> el.astype(np.int)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])

.. figure:: diamond_kernel.png
    :align: center

**侵蚀** = 最小值滤波，将::

    >>> a = np.zeros((7,7), dtype=np.int)
    >>> a[1:6, 2:5] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_erosion(a).astype(a.dtype)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> #Erosion removes objects smaller than the structure
    >>> ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])


.. image:: morpho_mat.png
    :align: center


**膨胀**: 最大值滤波::

    >>> a = np.zeros((5, 5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])


一些关于灰度图的应用::

    >>> np.random.seed(2)
    >>> im = np.zeros((64, 64))
    >>> x, y = (63*np.random.random((2, 8))).astype(np.int)
    >>> im[x, y] = np.arange(8)

    >>> bigger_points = ndimage.grey_dilation(im, size=(5, 5), structure=np.ones((5, 5)))

    >>> square = np.zeros((16, 16))
    >>> square[4:-4, 4:-4] = 1
    >>> dist = ndimage.distance_transform_bf(square)
    >>> dilate_dist = ndimage.grey_dilation(dist, size=(3, 3), \
    ...         structure=np.ones((3, 3)))


.. figure:: auto_examples/images/plot_greyscale_dilation_1.png
    :scale: 40
    :target: auto_examples/plot_greyscale_dilation.html

.. only:: html

    [:ref:`Python source code <example_plot_greyscale_dilation.py>`]

**开操作**: 腐蚀和膨胀::

    >>> a = np.zeros((5,5), dtype=np.int)
    >>> a[1:4, 1:4] = 1; a[4, 4] = 1
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 1]])
    >>> # Opening removes small objects
    >>> ndimage.binary_opening(a, structure=np.ones((3,3))).astype(np.int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Opening can also smooth corners
    >>> ndimage.binary_opening(a).astype(np.int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

**应用**: 去燥::

    >>> square = np.zeros((32, 32))
    >>> square[10:-10, 10:-10] = 1
    >>> np.random.seed(2)
    >>> x, y = (32*np.random.random((2, 20))).astype(np.int)
    >>> square[x, y] = 1

    >>> open_square = ndimage.binary_opening(square)

    >>> eroded_square = ndimage.binary_erosion(square)
    >>> reconstruction = ndimage.binary_propagation(eroded_square, mask=square)

.. figure:: auto_examples/images/plot_propagation_1.png
    :scale: 40
    :target: auto_examples/plot_propagation.html

.. only:: html

    [:ref:`Python source code <example_plot_propagation.py>`]

**闭合操作**: 膨胀和腐蚀

其它一些数学形态学的操作: 击中击不中变换，Top-hat变换，等。


特征提取
==================

边缘检测
--------------

生成数据::

    >>> im = np.zeros((256, 256))
    >>> im[64:-64, 64:-64] = 1
    >>>
    >>> im = ndimage.rotate(im, 15, mode='constant')
    >>> im = ndimage.gaussian_filter(im, 8)

U使用**梯度算符** (**Sobel**)来找到变化最大的区域::

    >>> sx = ndimage.sobel(im, axis=0, mode='constant')
    >>> sy = ndimage.sobel(im, axis=1, mode='constant')
    >>> sob = np.hypot(sx, sy)

.. figure:: auto_examples/images/plot_find_edges_1.png
    :scale: 40
    :target: auto_examples/plot_find_edges.html

.. only:: html

    [:ref:`Python source code <example_plot_find_edges.py>`]


分割
------------

* **基于直方图** 分割 (不含空间信息)

::

    >>> n = 10
    >>> l = 256
    >>> im = np.zeros((l, l))
    >>> np.random.seed(1)
    >>> points = l*np.random.random((2, n**2))
    >>> im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    >>> im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

    >>> mask = (im > im.mean()).astype(np.float)
    >>> mask += 0.1 * im
    >>> img = mask + 0.2*np.random.randn(*mask.shape)

    >>> hist, bin_edges = np.histogram(img, bins=60)
    >>> bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    >>> binary_img = img > 0.5

.. figure:: auto_examples/images/plot_histo_segmentation_1.png
    :scale: 65
    :target: auto_examples/plot_histo_segmentation.html

.. only:: html

    [:ref:`Python source code <example_plot_histo_segmentation.py>`]

实用数学形态学工具来清理结果::

    >>> # 去掉小白块
    >>> open_img = ndimage.binary_opening(binary_img)
    >>> # 去掉小黑块
    >>> close_img = ndimage.binary_closing(open_img)

.. figure:: auto_examples/images/plot_clean_morpho_1.png
    :scale: 65
    :target: auto_examples/plot_clean_morpho.html

.. only:: html

    [:ref:`Python source code <example_plot_clean_morpho.py>`]

.. topic:: **练习**
    :class: green

    查看重建操作 (腐蚀和传播) 来达到比开闭更好的效果::

	>>> eroded_img = ndimage.binary_erosion(binary_img)
	>>> reconstruct_img = ndimage.binary_propagation(eroded_img, mask=binary_img)
	>>> tmp = np.logical_not(reconstruct_img)
	>>> eroded_tmp = ndimage.binary_erosion(tmp)
	>>> reconstruct_final = np.logical_not(ndimage.binary_propagation(eroded_tmp, mask=tmp))
	>>> np.abs(mask - close_img).mean() # doctest: +ELLIPSIS
	0.00727836...
	>>> np.abs(mask - reconstruct_final).mean() # doctest: +ELLIPSIS
	0.00059502...

.. topic:: **练习**
    :class: green

    Check how a first denoising step (e.g. with a median filter)
    modifies the histogram, and check that the resulting histogram-based
    segmentation is more accurate.


.. seealso::

    其它一些关于分割的高级操作，可以参加
    ``scikit-image``: see :ref:`scikit_image`.

.. seealso::
    其它一些提供图像处理操作的包. 比如在这个例子中，我们使用``scikit-learn``中谱聚类的函数来分割胶着的物体。


    ::

        >>> from sklearn.feature_extraction import image
        >>> from sklearn.cluster import spectral_clustering

        >>> l = 100
        >>> x, y = np.indices((l, l))

        >>> center1 = (28, 24)
        >>> center2 = (40, 50)
        >>> center3 = (67, 58)
        >>> center4 = (24, 70)
        >>> radius1, radius2, radius3, radius4 = 16, 14, 15, 14

        >>> circle1 = (x - center1[0])**2 + (y - center1[1])**2 < radius1**2
        >>> circle2 = (x - center2[0])**2 + (y - center2[1])**2 < radius2**2
        >>> circle3 = (x - center3[0])**2 + (y - center3[1])**2 < radius3**2
        >>> circle4 = (x - center4[0])**2 + (y - center4[1])**2 < radius4**2

        >>> # 4个圆
        >>> img = circle1 + circle2 + circle3 + circle4
        >>> mask = img.astype(bool)
        >>> img = img.astype(float)

        >>> img += 1 + 0.2*np.random.randn(*img.shape)
        >>> # Convert the image into a graph with the value of the gradient on
        >>> # the edges.
        >>> graph = image.img_to_graph(img, mask=mask)

        >>> # Take a decreasing function of the gradient: we take it weakly
        >>> # dependant from the gradient the segmentation is close to a voronoi
        >>> graph.data = np.exp(-graph.data/graph.data.std())

        >>> labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
        >>> label_im = -np.ones(mask.shape)
        >>> label_im[mask] = labels


    .. image:: image_spectral_clustering.png
        :align: center



度量物体属性： ``ndimage.measurements``
========================================================

合成数据::

    >>> n = 10
    >>> l = 256
    >>> im = np.zeros((l, l))
    >>> points = l*np.random.random((2, n**2))
    >>> im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    >>> im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
    >>> mask = im > im.mean()

* **分析相连的部分**

给相连的部分打上标记: ``ndimage.label``::

    >>> label_im, nb_labels = ndimage.label(mask)
    >>> nb_labels # 有多少个不同的部分
    16
    >>> plt.imshow(label_im)        # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at 0x...>

.. figure:: auto_examples/images/plot_synthetic_data_1.png
    :scale: 90
    :target: auto_examples/plot_synthetic_data.html

.. only:: html

    [:ref:`Python source code <example_plot_synthetic_data.py>`]

计算每一个部分的大小和均值等等::

    >>> sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    >>> mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))

清除其中较小的一些::

    >>> mask_size = sizes < 1000
    >>> remove_pixel = mask_size[label_im]
    >>> remove_pixel.shape
    (256, 256)
    >>> label_im[remove_pixel] = 0
    >>> plt.imshow(label_im)        # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at 0x...>

现在通过``np.searchsorted``来贴上标签::

    >>> labels = np.unique(label_im)
    >>> label_im = np.searchsorted(labels, label_im)

.. figure:: auto_examples/images/plot_measure_data_1.png
    :scale: 90
    :target: auto_examples/plot_measure_data.html

.. only:: html

    [:ref:`Python source code <example_plot_measure_data.py>`]

找到包含所关注物体的区域::

    >>> slice_x, slice_y = ndimage.find_objects(label_im==4)[0]
    >>> roi = im[slice_x, slice_y]
    >>> plt.imshow(roi)     # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at 0x...>

.. figure:: auto_examples/images/plot_find_object_1.png
    :scale: 130
    :target: auto_examples/plot_find_object.html

.. only:: html

    [:ref:`Python source code <example_plot_find_object.py>`]

其它空间度量: ``ndimage.center_of_mass``,
``ndimage.maximum_position``, 等等.

也可用于除图像分割外的其它一些领域

例子: 块均值::

    >>> from scipy import misc
    >>> f = misc.face(gray=True)
    >>> sx, sy = f.shape
    >>> X, Y = np.ogrid[0:sx, 0:sy]
    >>> regions = (sy//6) * (X//4) + (Y//6)  # note that we use broadcasting
    >>> block_mean = ndimage.mean(f, labels=regions, index=np.arange(1,
    ...     regions.max() +1))
    >>> block_mean.shape = (sx // 4, sy // 6)

.. figure:: auto_examples/images/plot_block_mean_1.png
    :scale: 70
    :target: auto_examples/plot_block_mean.html

.. only:: html

    [:ref:`Python source code <example_plot_block_mean.py>`]

当区域是一个正方的块, 可以使用一个更高效的技巧(:ref:`stride-manipulation-label`).

非正方的区块: 径向均值::

    >>> sx, sy = f.shape
    >>> X, Y = np.ogrid[0:sx, 0:sy]
    >>> r = np.hypot(X - sx/2, Y - sy/2)
    >>> rbin = (20* r/r.max()).astype(np.int)
    >>> radial_mean = ndimage.mean(f, labels=rbin, index=np.arange(1, rbin.max() +1))

.. figure:: auto_examples/images/plot_radial_mean_1.png
    :scale: 70
    :target: auto_examples/plot_radial_mean.html

.. only:: html

    [:ref:`Python source code <example_plot_radial_mean.py>`]


* **其它度量方法**

相关函数, 傅立叶或者小波频谱，等等。

一个包含数学形态学的例子: `granulometry
<https://en.wikipedia.org/wiki/Granulometry_%28morphology%29>`_

::

    >>> def disk_structure(n):
    ...     struct = np.zeros((2 * n + 1, 2 * n + 1))
    ...     x, y = np.indices((2 * n + 1, 2 * n + 1))
    ...     mask = (x - n)**2 + (y - n)**2 <= n**2
    ...     struct[mask] = 1
    ...     return struct.astype(np.bool)
    ...
    >>>
    >>> def granulometry(data, sizes=None):
    ...     s = max(data.shape)
    ...     if sizes == None:
    ...         sizes = range(1, s/2, 2)
    ...     granulo = [ndimage.binary_opening(data, \
    ...         structure=disk_structure(n)).sum() for n in sizes]
    ...     return granulo
    ...
    >>>
    >>> np.random.seed(1)
    >>> n = 10
    >>> l = 256
    >>> im = np.zeros((l, l))
    >>> points = l*np.random.random((2, n**2))
    >>> im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    >>> im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
    >>>
    >>> mask = im > im.mean()
    >>>
    >>> granulo = granulometry(mask, sizes=np.arange(2, 19, 4))


.. figure:: auto_examples/images/plot_granulo_1.png
    :scale: 100
    :target: auto_examples/plot_granulo.html

.. only:: html

    [:ref:`Python source code <example_plot_granulo.py>`]

|


.. seealso:: 更多关于图像处理:

   *  :ref:`Scikit-image <scikit_image>`的有关章节
   
   * 更强大的图像处理工具: `OpenCV
     <https://opencv-python-tutroals.readthedocs.org/en/latest>`_
     , `CellProfiler <http://www.cellprofiler.org>`_,
     `ITK <http://www.itk.org/>`




