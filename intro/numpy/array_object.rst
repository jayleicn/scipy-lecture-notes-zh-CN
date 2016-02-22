..
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend("Agg")


.. currentmodule:: numpy

Numpy数组对象

.. contents:: Section contents
    :local:
    :depth: 1

什么是Numpy和Numpy数组?
--------------------------------

Numpy数组
............

:**Python** 内置对象:

    - 高级的数字对象: 整数(integer), 浮点数 (floating point)

    - 容器: 列表 (低开销的插入), 字典 (快速查找)

:**Numpy** 提供的对象:

    - 用于多维数组的第三方Python包

    - 更接近于底层和硬件 (高效)

    - 专注于科学计算 (方便)

    - 也被称作面向计算的数组

|

.. sourcecode:: pycon

    >>> import numpy as np
    >>> a = np.array([0, 1, 2, 3])
    >>> a
    array([0, 1, 2, 3])

.. tip::

    Numpy数组可以记录以下数据：

    * 离散事件仿真的数据变量

    * 设备测量的信号数据，如声波

    * 图像中每一个像素点的灰度值或颜色

    * 三维(多维)数据，如核磁共振扫描

    * ...

**为什么使用Numpy:** Numpy数组是一个较为节省内存的数据容器，并且其数值计算操作十分高效。

.. sourcecode:: ipython

    In [1]: L = range(1000)

    In [2]: %timeit [i**2 for i in L]
    1000 loops, best of 3: 403 us per loop

    In [3]: a = np.arange(1000)

    In [4]: %timeit a**2
    100000 loops, best of 3: 12.7 us per loop


.. extension package to Python to support multidimensional arrays

.. diagram, import conventions

.. scope of this tutorial: drill in features of array manipulation in
   Python, and try to give some indication on how to get things done
   in good style

.. a fixed number of elements (cf. certain exceptions)
.. each element of same size and type
.. efficiency vs. Python lists

Numpy参考文档
..............................

- 在线版: 请访问 http://docs.scipy.org/ 以获得帮助

- 交互式(如之前IPython说明提到的，直接在相关对象后加问号):

  .. sourcecode:: ipython

     In [5]: np.array?
     String Form:<built-in function array>
     Docstring:
     array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0, ...

  .. tip:

   .. sourcecode:: pycon

     >>> help(np.array) # doctest: +ELLIPSIS
     Help on built-in function array in module numpy.core.multiarray:
     <BLANKLINE>
     array(...)
         array(object, dtype=None, copy=True, order=None, subok=False, ...


- 查询某种结构或功能:

  .. sourcecode:: pycon

     >>> np.lookfor('create array') # doctest: +SKIP
     Search results for 'create array'
     ---------------------------------
     numpy.array
         Create an array.
     numpy.memmap
         Create a memory-map to an array stored in a *binary* file on disk.

  .. sourcecode:: ipython

     In [6]: np.con*?
     np.concatenate
     np.conj
     np.conjugate
     np.convolve

导入Numpy的建议
..................

我们推荐使用以下方法来导入Numpy:

.. sourcecode:: pycon

   >>> import numpy as np


创建Numpy数组
---------------

手动建立数组
..............................

* **一维数组**:

  .. sourcecode:: pycon

    >>> a = np.array([0, 1, 2, 3])
    >>> a
    array([0, 1, 2, 3])
    >>> a.ndim
    1
    >>> a.shape
    (4,)
    >>> len(a)
    4

* **二维及多维数组**:

  .. sourcecode:: pycon

    >>> b = np.array([[0, 1, 2], [3, 4, 5]])    # 2 x 3 array
    >>> b
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> b.ndim
    2
    >>> b.shape
    (2, 3)
    >>> len(b)     # returns the size of the first dimension
    2

    >>> c = np.array([[[1], [2]], [[3], [4]]])
    >>> c
    array([[[1],
            [2]],
    <BLANKLINE>
           [[3],
            [4]]])
    >>> c.shape
    (2, 2, 1)

.. topic:: **练习：简单的数组**
    :class: green

    * 创建一个简单的二维数组。 首先，重复一下上面的例子，然后创建一个你喜欢的数组。
    * 对你创建的数组使用len(), numpy.shape()函数，他们直接的关系是怎样的，再试着使用数组的ndim属性。

用函数创建数组
..............................

.. tip::

    事实上，我们很少手动一个一个地创建数组...

* 等间距分布的数组:

  .. sourcecode:: pycon

    >>> a = np.arange(10) # 0 .. n-1  (!)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = np.arange(1, 9, 2) # start, end (exclusive), step
    >>> b
    array([1, 3, 5, 7])

* 指定数量(长度)的数组:

  .. sourcecode:: pycon

    >>> c = np.linspace(0, 1, 6)   # start, end, num-points
    >>> c
    array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
    >>> d = np.linspace(0, 1, 5, endpoint=False)
    >>> d
    array([ 0. ,  0.2,  0.4,  0.6,  0.8])

* 一些常用的数组:

  .. sourcecode:: pycon

    >>> a = np.ones((3, 3))  # reminder: (3, 3) is a tuple
    >>> a
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])
    >>> b = np.zeros((2, 2))
    >>> b
    array([[ 0.,  0.],
           [ 0.,  0.]])
    >>> c = np.eye(3)
    >>> c
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> d = np.diag(np.array([1, 2, 3, 4]))
    >>> d
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

* :mod:`np.random`: 随机数数组(梅森旋转算法：大概和随机数生成有关系，不用在意细节...):

  .. sourcecode:: pycon

    >>> a = np.random.rand(4)       # uniform in [0, 1]
    >>> a  # doctest: +SKIP
    array([ 0.95799151,  0.14222247,  0.08777354,  0.51887998])

    >>> b = np.random.randn(4)      # Gaussian
    >>> b  # doctest: +SKIP
    array([ 0.37544699, -0.11425369, -0.47616538,  1.79664113])

    >>> np.random.seed(1234)        # Setting the random seed

.. topic:: **练习：使用函数创建数组**
   :class: green

   * 尝试使用 ``arange``, ``linspace``, ``ones``, ``zeros``, ``eye`` 和
     ``diag`` 函数创建数组。
   * 用随机数创建不同类型的数组。
   * 在使用随机数之前设置随机数种子(seed)。
   * 试试 ``np.empty`` 函数，看看它有什么用。

.. EXE: construct 1 2 3 4 5
.. EXE: construct -5, -4, -3, -2, -1
.. EXE: construct 2 4 6 8
.. EXE: look what is in an empty() array
.. EXE: construct 15 equispaced numbers in range [0, 10]

基本数据类型
----------------

你也许会发现，某些数组元素后面会跟着一个小数点 (比如 ``2.`` vs ``2``)。这是由于不同的数据类型所致:

.. sourcecode:: pycon

    >>> a = np.array([1, 2, 3])
    >>> a.dtype
    dtype('int64')

    >>> b = np.array([1., 2., 3.])
    >>> b.dtype
    dtype('float64')

.. tip::

    不同的数据结构可以让我们更加高效地使用内存，但是通常来说我们用浮点数就够了。
    在下面的例子中，Numpy会自动检测输入的数据类型。

-----------------------------

你可以显式地指定数据类型:

.. sourcecode:: pycon

    >>> c = np.array([1, 2, 3], dtype=float)
    >>> c.dtype
    dtype('float64')


默认的数据类型是浮点数:

.. sourcecode:: pycon

    >>> a = np.ones((3, 3))
    >>> a.dtype
    dtype('float64')

当然我们也有其他的数据类型:

:复数:

  .. sourcecode:: pycon

        >>> d = np.array([1+2j, 3+4j, 5+6*1j])
        >>> d.dtype
        dtype('complex128')

:布尔值:

  .. sourcecode:: pycon

        >>> e = np.array([True, False, False, True])
        >>> e.dtype
        dtype('bool')

:字符串:

  .. sourcecode:: pycon

        >>> f = np.array(['Bonjour', 'Hello', 'Hallo',])
        >>> f.dtype     # <--- strings containing max. 7 letters  # doctest: +SKIP
        dtype('S7')

:其他类型:

    * ``int32``
    * ``int64``
    * ``uint32``
    * ``uint64``

.. XXX: mention: astype


基本的数据可视化
-------------------

既然我们知道如何构造数组了，那么现在我们要试着将数组里的数据可视化。

首先在终端中打开IPython:

.. sourcecode:: bash

    $ ipython

当然你也可以使用IPython Notebook (我个人认为这个比较好用，还可以远程，只需要有一个浏览器):

.. sourcecode:: bash

   $ ipython notebook

打开IPython后，我们应该打开交互绘图的功能:

.. sourcecode:: pycon

    >>> %matplotlib  # doctest: +SKIP

如果你使用IPython Notebook的话，请以如下方式打开绘图功能:

.. sourcecode:: pycon

    >>> %matplotlib inline # doctest: +SKIP

这个 ``inline`` 属性可以让我们在notebook就能看到图像，而不会打开新的窗口。

*Matplotlib* 是一个2D的画图包，以如下方式导入：

.. sourcecode:: pycon

    >>> import matplotlib.pyplot as plt  # the tidy way

现在就可以使用了 (注意一下，如果你没有打开交互式画图功能的话，需要显式地调用 ``show`` 命令):

.. sourcecode:: pycon

    >>> plt.plot(x, y)       # line plot    # doctest: +SKIP
    >>> plt.show()           # <-- shows the plot (not needed with interactive plots) # doctest: +SKIP

当然，如果你执行了 ``%matplotlib`` ，那么一切会变得更加简单:

.. sourcecode:: pycon

    >>> plot(x, y)       # line plot    # doctest: +SKIP

* **一维图像**:

  .. sourcecode:: pycon

    >>> x = np.linspace(0, 3, 20)
    >>> y = np.linspace(0, 9, 20)
    >>> plt.plot(x, y)       # line plot    # doctest: +SKIP
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(x, y, 'o')  # dot plot    # doctest: +SKIP
    [<matplotlib.lines.Line2D object at ...>]

  .. plot:: pyplots/numpy_intro_1.py

* **二维图像** (比如常见的图片什么的):

  .. sourcecode:: pycon

    >>> image = np.random.rand(30, 30)
    >>> plt.imshow(image, cmap=plt.cm.hot)    # doctest: +SKIP
    >>> plt.colorbar()    # doctest: +SKIP
    <matplotlib.colorbar.Colorbar instance at ...>

  .. plot:: pyplots/numpy_intro_2.py

.. seealso:: 更多请见: :ref:`matplotlib chapter <matplotlib>`

.. topic:: **练习：简单的数据可视化**
   :class: green

   * 画出一些简单的数组: 一个关于时间的cos函数和二维矩阵。
   * 尝试在一个二维矩阵上使用 ``gray`` 颜色图(灰度图)。

.. * **3D plotting**:
..
..   For 3D visualization, we can use another package: **Mayavi**. A quick example:
..   start by **relaunching iPython** with these options: **ipython --pylab=wx**
..   (or **ipython -pylab -wthread** in IPython < 0.10).
..
..   .. image:: surf.png
..      :align: right
..      :scale: 60
..
..   .. sourcecode:: ipython
..
..       In [58]: from mayavi import mlab
..       In [61]: mlab.surf(image)
..       Out[61]: <enthought.mayavi.modules.surface.Surface object at ...>
..       In [62]: mlab.axes()
..       Out[62]: <enthought.mayavi.modules.axes.Axes object at ...>
..
..   .. tip::
..
..    The mayavi/mlab window that opens is interactive: by clicking on the
..    left mouse button you can rotate the image, zoom with the mouse wheel,
..    etc.
..
..    For more information on Mayavi :
..    https://github.enthought.com/mayavi/mayavi
..
..   .. seealso:: More in the :ref:`Mayavi chapter <mayavi-label>`


Indexing and slicing
--------------------

The items of an array can be accessed and assigned to the same way as
other Python sequences (e.g. lists):

.. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> a[0], a[2], a[-1]
    (0, 2, 9)

.. warning::

   Indices begin at 0, like other Python sequences (and C/C++).
   In contrast, in Fortran or Matlab, indices begin at 1.

The usual python idiom for reversing a sequence is supported:

.. sourcecode:: pycon

   >>> a[::-1]
   array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

For multidimensional arrays, indexes are tuples of integers:

.. sourcecode:: pycon

    >>> a = np.diag(np.arange(3))
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 2]])
    >>> a[1, 1]
    1
    >>> a[2, 1] = 10 # third line, second column
    >>> a
    array([[ 0,  0,  0],
           [ 0,  1,  0],
           [ 0, 10,  2]])
    >>> a[1]
    array([0, 1, 0])


.. note::

  * In 2D, the first dimension corresponds to **rows**, the second
    to **columns**.
  * for multidimensional ``a``, ``a[0]`` is interpreted by
    taking all elements in the unspecified dimensions.

**Slicing**: Arrays, like other Python sequences can also be sliced:

.. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> a[2:9:3] # [start:end:step]
    array([2, 5, 8])

Note that the last index is not included! :

.. sourcecode:: pycon

    >>> a[:4]
    array([0, 1, 2, 3])

All three slice components are not required: by default, `start` is 0,
`end` is the last and `step` is 1:

.. sourcecode:: pycon

    >>> a[1:3]
    array([1, 2])
    >>> a[::2]
    array([0, 2, 4, 6, 8])
    >>> a[3:]
    array([3, 4, 5, 6, 7, 8, 9])

A small illustrated summary of Numpy indexing and slicing...

.. only:: latex

    .. image:: images/numpy_indexing.png
        :align: center

.. only:: html

    .. image:: images/numpy_indexing.png
        :align: center
        :width: 70%

You can also combine assignment and slicing:

.. sourcecode:: pycon

   >>> a = np.arange(10)
   >>> a[5:] = 10
   >>> a
   array([ 0,  1,  2,  3,  4, 10, 10, 10, 10, 10])
   >>> b = np.arange(5)
   >>> a[5:] = b[::-1]
   >>> a
   array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0])

.. topic:: **Exercise: Indexing and slicing**
   :class: green

   * Try the different flavours of slicing, using ``start``, ``end`` and
     ``step``: starting from a linspace, try to obtain odd numbers
     counting backwards, and even numbers counting forwards.
   * Reproduce the slices in the diagram above. You may
     use the following expression to create the array:

     .. sourcecode:: pycon

        >>> np.arange(6) + np.arange(0, 51, 10)[:, np.newaxis]
        array([[ 0,  1,  2,  3,  4,  5],
               [10, 11, 12, 13, 14, 15],
               [20, 21, 22, 23, 24, 25],
               [30, 31, 32, 33, 34, 35],
               [40, 41, 42, 43, 44, 45],
               [50, 51, 52, 53, 54, 55]])

.. topic:: **Exercise: Array creation**
    :class: green

    Create the following arrays (with correct data types)::

        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 2],
         [1, 6, 1, 1]]

        [[0., 0., 0., 0., 0.],
         [2., 0., 0., 0., 0.],
         [0., 3., 0., 0., 0.],
         [0., 0., 4., 0., 0.],
         [0., 0., 0., 5., 0.],
         [0., 0., 0., 0., 6.]]

    Par on course: 3 statements for each

    *Hint*: Individual array elements can be accessed similarly to a list,
    e.g. ``a[1]`` or ``a[1, 2]``.

    *Hint*: Examine the docstring for ``diag``.

.. topic:: Exercise: Tiling for array creation
    :class: green

    Skim through the documentation for ``np.tile``, and use this function
    to construct the array::

        [[4, 3, 4, 3, 4, 3],
         [2, 1, 2, 1, 2, 1],
         [4, 3, 4, 3, 4, 3],
         [2, 1, 2, 1, 2, 1]]

Copies and views
----------------

A slicing operation creates a **view** on the original array, which is
just a way of accessing array data. Thus the original array is not
copied in memory. You can use ``np.may_share_memory()`` to check if two arrays
share the same memory block. Note however, that this uses heuristics and may
give you false positives.

**When modifying the view, the original array is modified as well**:

.. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = a[::2]
    >>> b
    array([0, 2, 4, 6, 8])
    >>> np.may_share_memory(a, b)
    True
    >>> b[0] = 12
    >>> b
    array([12,  2,  4,  6,  8])
    >>> a   # (!)
    array([12,  1,  2,  3,  4,  5,  6,  7,  8,  9])

    >>> a = np.arange(10)
    >>> c = a[::2].copy()  # force a copy
    >>> c[0] = 12
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    >>> np.may_share_memory(a, c)
    False



This behavior can be surprising at first sight... but it allows to save both
memory and time.


.. EXE: [1, 2, 3, 4, 5] -> [1, 2, 3]
.. EXE: [1, 2, 3, 4, 5] -> [4, 5]
.. EXE: [1, 2, 3, 4, 5] -> [1, 3, 5]
.. EXE: [1, 2, 3, 4, 5] -> [2, 4]
.. EXE: create an array [1, 1, 1, 1, 0, 0, 0]
.. EXE: create an array [0, 0, 0, 0, 1, 1, 1]
.. EXE: create an array [0, 1, 0, 1, 0, 1, 0]
.. EXE: create an array [1, 0, 1, 0, 1, 0, 1]
.. EXE: create an array [1, 0, 2, 0, 3, 0, 4]
.. CHA: archimedean sieve

.. topic:: Worked example: Prime number sieve
   :class: green

   .. image:: images/prime-sieve.png

   Compute prime numbers in 0--99, with a sieve

   * Construct a shape (100,) boolean array ``is_prime``,
     filled with True in the beginning:

   .. sourcecode:: pycon

        >>> is_prime = np.ones((100,), dtype=bool)

   * Cross out 0 and 1 which are not primes:

   .. sourcecode:: pycon

       >>> is_prime[:2] = 0

   * For each integer ``j`` starting from 2, cross out its higher multiples:

   .. sourcecode:: pycon

       >>> N_max = int(np.sqrt(len(is_prime)))
       >>> for j in range(2, N_max):
       ...     is_prime[2*j::j] = False

   * Skim through ``help(np.nonzero)``, and print the prime numbers

   * Follow-up:

     - Move the above code into a script file named ``prime_sieve.py``

     - Run it to check it works

     - Use the optimization suggested in `the sieve of Eratosthenes
       <https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes>`_:

      1. Skip ``j`` which are already known to not be primes

      2. The first number to cross out is :math:`j^2`

Fancy indexing
--------------

.. tip::

    Numpy arrays can be indexed with slices, but also with boolean or
    integer arrays (**masks**). This method is called *fancy indexing*.
    It creates **copies not views**.

Using boolean masks
...................

.. sourcecode:: pycon

    >>> np.random.seed(3)
    >>> a = np.random.random_integers(0, 20, 15)
    >>> a
    array([10,  3,  8,  0, 19, 10, 11,  9, 10,  6,  0, 20, 12,  7, 14])
    >>> (a % 3 == 0)
    array([False,  True, False,  True, False, False, False,  True, False,
            True,  True, False,  True, False, False], dtype=bool)
    >>> mask = (a % 3 == 0)
    >>> extract_from_a = a[mask] # or,  a[a%3==0]
    >>> extract_from_a           # extract a sub-array with the mask
    array([ 3,  0,  9,  6,  0, 12])

Indexing with a mask can be very useful to assign a new value to a sub-array:

.. sourcecode:: pycon

    >>> a[a % 3 == 0] = -1
    >>> a
    array([10, -1,  8, -1, 19, 10, 11, -1, 10, -1, -1, 20, -1,  7, 14])


Indexing with an array of integers
..................................

.. sourcecode:: pycon

    >>> a = np.arange(0, 100, 10)
    >>> a
    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

Indexing can be done with an array of integers, where the same index is repeated
several time:

.. sourcecode:: pycon

    >>> a[[2, 3, 2, 4, 2]]  # note: [2, 3, 2, 4, 2] is a Python list
    array([20, 30, 20, 40, 20])

New values can be assigned with this kind of indexing:

.. sourcecode:: pycon

    >>> a[[9, 7]] = -100
    >>> a
    array([   0,   10,   20,   30,   40,   50,   60, -100,   80, -100])

.. tip::

  When a new array is created by indexing with an array of integers, the
  new array has the same shape than the array of integers:

  .. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> idx = np.array([[3, 4], [9, 7]])
    >>> idx.shape
    (2, 2)
    >>> a[idx]
    array([[3, 4],
           [9, 7]])


____

The image below illustrates various fancy indexing applications

.. only:: latex

    .. image:: images/numpy_fancy_indexing.png
        :align: center

.. only:: html

    .. image:: images/numpy_fancy_indexing.png
        :align: center
        :width: 80%

.. topic:: **Exercise: Fancy indexing**
    :class: green

    * Again, reproduce the fancy indexing shown in the diagram above.
    * Use fancy indexing on the left and array creation on the right to assign
      values into an array, for instance by setting parts of the array in
      the diagram above to zero.

.. We can even use fancy indexing and :ref:`broadcasting <broadcasting>` at
.. the same time:
..
.. .. sourcecode:: pycon
..
..     >>> a = np.arange(12).reshape(3,4)
..     >>> a
..     array([[ 0,  1,  2,  3],
..            [ 4,  5,  6,  7],
..            [ 8,  9, 10, 11]])
..     >>> i = np.array([[0, 1], [1, 2]])
..     >>> a[i, 2] # same as a[i, 2*np.ones((2, 2), dtype=int)]
..     array([[ 2,  6],
..            [ 6, 10]])
