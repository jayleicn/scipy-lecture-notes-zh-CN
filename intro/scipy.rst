.. for doctests
    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend("Agg")
    >>> import numpy as np
    >>> np.random.seed(0)

.. _scipy:

Scipy : 高性能科学计算
=======================================

**作者**: *Adrien Chauve, Andre Espaze, Emmanuelle Gouillart, Gaël
Varoquaux, Ralf Gommers*

**译者**: *Jay Lei*


.. topic:: Scipy

    ``scipy`` 中有许多科学计算常见问题的工具，比如内置了图像处理，
    优化，统计等等相关问题的子模块。

    正如GSL库在C和C++中，或者相关工具箱在Matlab中的地位一样， ``scipy`` 是Python科学计算环境的核心。
    它被设计为利用 ``numpy`` 数组进行高效的运行。从这个角度来讲，scipy和numpy是密不可分的。

    在执行一段程序前，很有必要检查数据处理是否是已经用Scipy完成了。不专业的程序员或者科学家们经常
    会去 **重新造轮子** 。 多数时候这并不是一件好事，相比于Scipy中优化过的程序，这些新的“轮子”通常
    存在缺陷，未完全优化，以及可维护性差，不易分享等等问题。


.. contents:: Chapters contents
    :local:
    :depth: 1


.. warning::

    这份教程的目的并不是提供一个数值计算的简介。因逐一列举式地去介绍scipy中的各个模块和
    函数会显得十分无趣，我们将重点放在了介绍一些能够说明 ``scipy`` 在科学计算中的作用的
    小例子。

:mod:`scipy` 由一些处理不同任务的子模块构成:

=========================== ==========================================
:mod:`scipy.cluster`         矢量量化 / Kmeans
:mod:`scipy.constants`       数学，物理常量
:mod:`scipy.fftpack`         傅里叶变换
:mod:`scipy.integrate`       Integration routines
:mod:`scipy.interpolate`     插值计算
:mod:`scipy.io`              数据输入输出
:mod:`scipy.linalg`          线性代数程序
:mod:`scipy.ndimage`         n维图像处理包
:mod:`scipy.odr`             正交距离回归
:mod:`scipy.optimize`        优化
:mod:`scipy.signal`          信号处理
:mod:`scipy.sparse`          稀疏矩阵
:mod:`scipy.spatial`         空间数据结果和算法
:mod:`scipy.special`         特殊数学函数
:mod:`scipy.stats`           统计
=========================== ==========================================

.. tip::
   
   所有的这些子模块都构建在 :mod:`numpy` 基础之上, 但是它们之间绝大部分是相互独立的
   引入Numpy和Scipy中这些小模块的标准方式如下所示
   ::

    >>> import numpy as np
    >>> from scipy import stats  # same for other sub-modules

   ``scipy`` 命名空间中直接包含的函数多是真实的numpy函数
   (比如 ``scipy.cos 是 np.cos``)。因此，一般来讲你的程序中不会用到 ``import
   scipy`` 。


文件输入输出: :mod:`scipy.io`
----------------------------------

* 载入和保存matlab文件::

    >>> from scipy import io as spio
    >>> a = np.ones((3, 3))
    >>> spio.savemat('file.mat', {'a': a}) # savemat expects a dictionary
    >>> data = spio.loadmat('file.mat', struct_as_record=True)
    >>> data['a']
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])

.. Comments to make doctests pass which require an image
    >>> from matplotlib import pyplot as plt
    >>> plt.imsave('fname.png', np.array([[0]]))

* 读取图像::

    >>> from scipy import misc
    >>> misc.imread('fname.png')    # doctest: +ELLIPSIS
    array(...)
    >>> # Matplotlib中有一个相似的函数
    >>> import matplotlib.pyplot as plt
    >>> plt.imread('fname.png')    # doctest: +ELLIPSIS
    array(...)

参见:

    * 载入文本文件: :func:`numpy.loadtxt`/:func:`numpy.savetxt`

    * 格式化载入text/csv文件:
      :func:`numpy.genfromtxt`/:func:`numpy.recfromcsv`

    * 高效快速载入numpy指定类型的，二进制格式文件:
      :func:`numpy.save`/:func:`numpy.load`


特殊函数: :mod:`scipy.special`
---------------------------------------

这里的特殊函数指的是超越函数。 :mod:`scipy.special` 模块的文档写得很详细，在此
我们仅列出一些常用的特殊函数:

 * Bessel函数, 比如 :func:`scipy.special.jn` (n阶Bessel函数)

 * 椭圆函数 (Jacobian椭圆函数 :func:`scipy.special.ellipj` )

 * Gamma函数: :func:`scipy.special.gamma`, 另有
   :func:`scipy.special.gammaln` 对数形式给出的精确度更高的Gamma函数。

 * Erf, 高斯曲线下方的面积: :func:`scipy.special.erf`


.. _scipy_linalg:

线性代数操作: :mod:`scipy.linalg`
----------------------------------------------

:mod:`scipy.linalg` 模块提供了基于BLAS和LAPACK的高效的代数操作方法。

* :func:`scipy.linalg.det` 计算方阵的行列式::

    >>> from scipy import linalg
    >>> arr = np.array([[1, 2],
    ...                 [3, 4]])
    >>> linalg.det(arr)
    -2.0
    >>> arr = np.array([[3, 2],
    ...                 [6, 4]])
    >>> linalg.det(arr)
    0.0
    >>> linalg.det(np.ones((3, 4)))
    Traceback (most recent call last):
    ...
    ValueError: expected square matrix

* :func:`scipy.linalg.inv` 计算方阵的逆矩阵::

    >>> arr = np.array([[1, 2],
    ...                 [3, 4]])
    >>> iarr = linalg.inv(arr)
    >>> iarr
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    >>> np.allclose(np.dot(arr, iarr), np.eye(2))
    True

  如果计算奇异矩阵(其行列式为0)的逆矩阵，函数会抛出 ``LinAlgError``::

    >>> arr = np.array([[3, 2],
    ...                 [6, 4]])
    >>> linalg.inv(arr)  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    ...LinAlgError: singular matrix

* 奇异值分解(SVD)::

    >>> arr = np.arange(9).reshape((3, 3)) + np.diag([1, 0, 1])
    >>> uarr, spec, vharr = linalg.svd(arr)

  奇异值如下所示::

    >>> spec    # doctest: +ELLIPSIS
    array([ 14.88982544,   0.45294236,   0.29654967])

  原始的矩阵可以使用 ``svd`` 的输出结果和 ``np.dot`` 的乘积重新生成::

    >>> sarr = np.diag(spec)
    >>> svd_mat = uarr.dot(sarr).dot(vharr)
    >>> np.allclose(svd_mat, arr)
    True

  SVD常用于统计和信号处理领域。其他的一些标准分解方法(QR, LU, Cholesky, Schur)
  在 :mod:`scipy.linalg` 中也能够找到。


快速傅里叶变换: :mod:`scipy.fftpack`
---------------------------------------------

 :mod:`scipy.fftpack` 模块包含了快速傅里叶变换的功能.
下面是一个噪声信号的例子::

    >>> time_step = 0.02
    >>> period = 5.
    >>> time_vec = np.arange(0, 20, time_step)
    >>> sig = np.sin(2 * np.pi / period * time_vec) + \
    ...       0.5 * np.random.randn(time_vec.size)

观察者不知道信号的频率，只知道信号的采样间隙 ``sig``. 信号是来自真实函数的，那么
傅里叶变换是对称的。 :func:`scipy.fftpack.fftfreq` 
函数会生成采样频率，:func:`scipy.fftpack.fft` 则用于进行快速傅里叶变化::

    >>> from scipy import fftpack
    >>> sample_freq = fftpack.fftfreq(sig.size, d=time_step)
    >>> sig_fft = fftpack.fft(sig)

Because the resulting power is symmetric, only the positive part of the
spectrum needs to be used for finding the frequency::

    >>> pidxs = np.where(sample_freq > 0)
    >>> freqs = sample_freq[pidxs]
    >>> power = np.abs(sig_fft)[pidxs]

.. plot:: pyplots/fftpack_frequency.py
    :scale: 70

信号频率可通过如下方式获得::

    >>> freq = freqs[power.argmax()]
    >>> np.allclose(freq, 1./period)  # check that correct freq is found
    True

滤去傅里叶变化后的信号中的高频噪声::

    >>> sig_fft[np.abs(sample_freq) > freq] = 0

去噪后信号可通过如下方式计算：
:func:`scipy.fftpack.ifft` function::

    >>> main_sig = fftpack.ifft(sig_fft)

结果如下::

    >>> import pylab as plt
    >>> plt.figure()    # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at 0x...>
    >>> plt.plot(time_vec, sig)    # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(time_vec, main_sig, linewidth=3)    # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.xlabel('Time [s]')    # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.ylabel('Amplitude')    # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>

.. plot:: pyplots/fftpack_signals.py
    :scale: 70

.. topic:: `numpy.fft`

   Numpy也有一个FFT的子模块 (:mod:`numpy.fft`). 但是Scipy中的FFT更为常用，
   因其底层实现更为高效

.. topic:: 例子: 寻找自然周期

    .. plot:: intro/solutions/periodicity_finder.py

.. topic:: 例子: 高斯模糊

    卷积:

    .. math::

        f_1(t) = \int dt'\, K(t-t') f_0(t')

    .. math::

        \tilde{f}_1(\omega) = \tilde{K}(\omega) \tilde{f}_0(\omega)

    .. plot:: intro/solutions/image_blur.py

.. topic:: 练习: 去除登月图像中的噪点
   :class: green

   .. image:: ../data/moonlanding.png
     :scale: 70

   1. 如图，moonlanding.png被周期性的噪点所污染。在这个练习中，我们将会使用快速傅里叶变化来进行去噪。

   2. 使用 :func:`pylab.imread` 函数读取图像。

   3. 使用 :mod:`scipy.fftpack` 模块中的2维傅里叶变换函数, 画出图像的频谱。不能可视化频谱？哪里出了问题？

   4. 频谱包含高频和低频成分。而噪声包含在高频部分，通过将这些部分的值有选择性的置为0可达到去噪的目的(利用数组切片特性).

   5. 使用傅里叶反变换得到图像。


优化和拟合: :mod:`scipy.optimize`
-------------------------------------------

优化用于求解最小化或者等式一类问题的数值解。

 :mod:`scipy.optimize` 模块提供了求解函数最小值，曲线拟合等算法的实现。
 ::

    >>> from scipy import optimize


**求解标量函数的最小值**

定义下面的函数: ::

    >>> def f(x):
    ...     return x**2 + 10*np.sin(x)

画出其图像:

.. doctest::

    >>> x = np.arange(-10, 10, 0.1)
    >>> plt.plot(x, f(x)) # doctest:+SKIP
    >>> plt.show() # doctest:+SKIP

.. plot:: pyplots/scipy_optimize_example1.py

此函数有一个全局最小值，约为-1.3， 还有一个局部最小值，约为3.8。

一个常用的求解此函数最小值的方法是确定初始点，然后执行梯度下降算法。BFGS算法是一个
很好的适用于此的方法::

    >>> optimize.fmin_bfgs(f, 0)
    Optimization terminated successfully.
	     Current function value: -7.945823
	     Iterations: 5
	     Function evaluations: 24
	     Gradient evaluations: 8
    array([-1.30644003])

这个方法的缺陷在于有时候可能会被困在一个局部最小值，而得不到全局的最小值。
这取决与初始点的选取: ::

    >>> optimize.fmin_bfgs(f, 3, disp=0)
    array([ 3.83746663])

如果我们不知道全局最小值的邻近数值，就需要使用那些可以实现全局最优化的算法。比如 :func:`scipy.optimize.basinhopping` 
包含一个求解局部最小值的算法和一个为该算法提供随机初始点的函数:


.. Comment to make doctest pass
   >>> np.random.seed(42)

.. versionadded:: 0.12.0 basinhopping was added in version 0.12.0 of Scipy

::

   >>> optimize.basinhopping(f, 0)  # doctest: +SKIP
                     nfev: 1725
    minimization_failures: 0
                      fun: -7.9458233756152845
                        x: array([-1.30644001])
                  message: ['requested number of basinhopping iterations completed successfully']
                     njev: 575
                      nit: 100

另外一个可用的，但不怎么高效的全局最优化算法是
:func:`scipy.optimize.brute` 。在 ``scipy`` 中包含的算法之外，还有许多可以实现全局最优化的算法。
这里使一些拥有相关算法的包OpenOpt, IPOPT_, PyGMO_ and PyEvolve_。

.. note::

   在老版本的 ``scipy`` 中，还包含 `退火` 算法。

.. _IPOPT: https://github.com/xuy/pyipopt
.. _PyGMO: http://esa.github.io/pygmo/
.. _PyEvolve: http://pyevolve.sourceforge.net/

为了找到局部最小值，可以把变量限制在区间``(0, 10)`` 中，
使用 :func:`scipy.optimize.fminbound`: ::

    >>> xmin_local = optimize.fminbound(f, 0, 10)
    >>> xmin_local    # doctest: +ELLIPSIS
    3.8374671...

.. note::

   寻找函数的最小值在更高级的章节中有提到: :ref:`mathematical_optimization`.

**寻找标量函数的零点**

例如，求解 :math:`f(x) = 0` 的零点, 其中 :math:`f` 是我们在上面用到的函数。
可以使用 :func:`scipy.optimize.fsolve`: ::

    >>> root = optimize.fsolve(f, 1)  # our initial guess is 1
    >>> root
    array([ 0.])

从上面的图像中我们可以看出函数 :math:`f` 包含两个零点。第二个零点在-2.5附近。
通过调整初始值，我们可以找出精确解: ::

    >>> root2 = optimize.fsolve(f, -2.5)
    >>> root2
    array([-2.47948183])

**曲线拟合**

.. Comment to make doctest pass
    >>> np.random.seed(42)

假设我们现在有从函数 :math:`f` 中采样得到的含有一些噪声的数据: ::


    >>> xdata = np.linspace(-10, 10, num=20)
    >>> ydata = f(xdata) + np.random.randn(xdata.size)

我们已经知道了函数的形式是 (:math:`x^2 + \sin(x)` ) 但不知道每一项系数的大小。
我们可以使用最小二乘算法来进行曲线拟合得到系数的值。 首先定义需要进行拟合的函数::

    >>> def f2(x, a, b):
    ...     return a*x**2 + b*np.sin(x)

T接着使用 :func:`scipy.optimize.curve_fit` 来求解 :math:`a` 和 :math:`b`: ::

    >>> guess = [2, 2]
    >>> params, params_covariance = optimize.curve_fit(f2, xdata, ydata, guess)
    >>> params
    array([  0.99667386,  10.17808313])

现在我们已经找到了函数 ``f`` 的最小值和零点，并且对采自这个函数的数据进行了曲线拟合的实验。
我们可以把所有的结果呈现在同一张图像上:

.. plot:: pyplots/scipy_optimize_example2.py

.. note::

   在 Scipy >= 0.11 的版本中，求解最小值和零点的函数可以通过:
   :func:`scipy.optimize.minimize`,
   :func:`scipy.optimize.minimize_scalar`,
   :func:`scipy.optimize.root`。 并且允许使用 ``method`` 关键字比较不同的算法.

你可以在 :mod:`scipy.optimize` 中找到适用于多维函数的算法.

.. topic:: 练习: 对温度数据进行曲线拟合。
   :class: green

    阿拉斯加每月的气温变化都很剧烈, 下面是从一月开始，阿拉斯加每月的温度情况
    (单位为摄氏度)::

        最大值:  17,  19,  21,  28,  33,  38, 37,  37,  31,  23,  19,  18
        最小值: -62, -59, -56, -46, -32, -18, -9, -13, -25, -46, -52, -58

    1. 画出温度图像
    2. 定义一个可以描述温度最小值和最大值的函数。
       提示: 这个函数的周期为一年。
       提示: 包含一个时间偏置量。
    3. 使用 :func:`scipy.optimize.curve_fit` 来拟合函数。
    4. 画出函数图象。拟合的结果是否合理?  如果不合理，为什么?
    5. 在拟合精度范围内，温度最大值和最小值的时间偏置量是否一样?

.. topic:: 联系: 2-D 最小值求解
   :class: green

    .. plot:: pyplots/scipy_optimize_sixhump.py

    驼峰函数

    .. math:: f(x, y) = (4 - 2.1x^2 + \frac{x^4}{3})x^2 + xy + (4y^2 - 4)y^2

    有多个全局和局部最小值。现在需要寻找此函数的全局最小值。

    提示:

        - 限制变量范围 :math:`-2 < x < 2` ， :math:`-1 < y < 1`。
        - 使用 :func:`numpy.meshgrid` 和 :func:`pylab.imshow` 目测最小值所在区域。
        - 使用 :func:`scipy.optimize.fmin_bfgs` 或者其他的用于可以求解多维函数最小值的算法

    函数有多少个全局最小值点，其对应的函数值是多少？如果初始点为 :math:`(x, y) = (0, 0)` 会怎么样?

更多的例子:ref:`summary_exercise_optimize` 。


统计和随机数: :mod:`scipy.stats`
-------------------------------------------------

 :mod:`scipy.stats` 包含一些统计和随机过程相关的工具。在 :mod:`numpy.random` 
 中可以找到生成多种随机过程的随机数生成器。

直方图和概率密度函数
..........................................

给出随机过程的一系列观察点，它们的直方图即为随机过程概率密度函数的一个估计: ::

    >>> a = np.random.normal(size=1000)
    >>> bins = np.arange(-4, 5)
    >>> bins
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> histogram = np.histogram(a, bins=bins, normed=True)[0]
    >>> bins = 0.5*(bins[1:] + bins[:-1])
    >>> bins
    array([-3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5])
    >>> from scipy import stats
    >>> b = stats.norm.pdf(bins)  # norm is a distribution
    
    >>> plt.plot(bins, histogram) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(bins, b) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]

.. plot:: pyplots/normal_distribution.py
    :scale: 70

如果我们事先已经知道所要处理的随机过程属于某一个随机过程族(比如正态过程)，可以采用最大似然估计来
得到相关参数的值。下面是估计正态过程来拟合观察点的例子::

    >>> loc, std = stats.norm.fit(a)
    >>> loc     # doctest: +ELLIPSIS
    0.0314345570...
    >>> std     # doctest: +ELLIPSIS
    0.9778613090...

.. topic:: 练习: 概率分布
   :class: green

   从形状参数为1的Gamma分布中取出1000个随机数值，画出它们的直方图。你能在其上画出原分布的
   概率密度函数图像吗(它们应该是相似的)？

   特别地: 你能根据这些数据点得出Gamma分布的形状参数值吗(使用 ``fit`` 方法)？


百分位数
...........

中位数::

    >>> np.median(a)     # doctest: +ELLIPSIS
    0.04041769593...

也叫第50百分位数::

    >>> stats.scoreatpercentile(a, 50)     # doctest: +ELLIPSIS
    0.0404176959...

相似的，我们可以计算第90百分位数::

    >>> stats.scoreatpercentile(a, 90)     # doctest: +ELLIPSIS
    1.3185699120...

百分位数是累计概率分布函数的一个估计。

统计检验
.................

统计检验的结果常用作一个决策指标。例如，如果我们有两组观察点，它们都来自高斯过程，我们可以使用
`T-检验 <https://en.wikipedia.org/wiki/Student%27s_t-test>`__ 来判断两组观察点是都显著不同::

    >>> a = np.random.normal(0, 1, size=100)
    >>> b = np.random.normal(1, 1, size=10)
    >>> stats.ttest_ind(a, b)   # doctest: +SKIP
    (array(-3.177574054...), 0.0019370639...)

.. tip:: 输出结果由以下部分组成:

    * T统计量的值: it is a number the sign of which is
      proportional to the difference between the two random processes and
      the magnitude is related to the significance of this difference.

    *  *p 值*: 两个过程相同的概率。如果其值接近1，那么两个过程几乎可以确定是相同的，如果其值接近0，
      那么它们很可能拥有不同的均值。

.. 参见::

    :ref:`statistics <statistics>` 章节介绍了许多scipy之外的关于统计检验及可视化等等的工具。


插值计算: :mod:`scipy.interpolate`
---------------------------------------

 :mod:`scipy.interpolate` 模块在拟合实验数据并估计未知点数值方面非常有用。
 这个模块是基于来自 netlib_ 项目的 `FITPACK Fortran subroutines`_ 。

.. _`FITPACK Fortran subroutines` : http://www.netlib.org/dierckx/index.html
.. _netlib : http://www.netlib.org

产生一个近似正弦函数的一系列实验数据::

    >>> measured_time = np.linspace(0, 1, 10)
    >>> noise = (np.random.random(10)*2 - 1) * 1e-1
    >>> measures = np.sin(2 * np.pi * measured_time) + noise

 :class:`scipy.interpolate.interp1d` 类可以创建一个线性插值函数::

    >>> from scipy.interpolate import interp1d
    >>> linear_interp = interp1d(measured_time, measures)

 :obj:`scipy.interpolate.linear_interp` 实例可以在需要的时候获取某些值::

    >>> computed_time = np.linspace(0, 1, 50)
    >>> linear_results = linear_interp(computed_time)

三次插值函数可通过 ``kind`` 关键字参数得到::

    >>> cubic_interp = interp1d(measured_time, measures, kind='cubic')
    >>> cubic_results = cubic_interp(computed_time)

所有结果呈现在如下的Matplotlib图像中:

.. plot:: pyplots/scipy_interpolation.py

:class:`scipy.interpolate.interp2d` 和
:class:`scipy.interpolate.interp1d` 较为相似, 但是其适用对象为二维数组。
获取样条插值的使用实例 :ref:`summary_exercise_stat_interp` 。 


Numerical integration: :mod:`scipy.integrate`
---------------------------------------------

The most generic integration routine is :func:`scipy.integrate.quad`::

    >>> from scipy.integrate import quad
    >>> res, err = quad(np.sin, 0, np.pi/2)
    >>> np.allclose(res, 1)
    True
    >>> np.allclose(err, 1 - res)
    True

Others integration schemes are available with ``fixed_quad``,
``quadrature``, ``romberg``.

:mod:`scipy.integrate` also features routines for integrating Ordinary
Differential Equations (ODE). In particular, :func:`scipy.integrate.odeint`
is a general-purpose integrator using LSODA (Livermore Solver for
Ordinary Differential equations with Automatic method switching
for stiff and non-stiff problems), see the `ODEPACK Fortran library`_
for more details.

.. _`ODEPACK Fortran library` : http://people.sc.fsu.edu/~jburkardt/f77_src/odepack/odepack.html

``odeint`` solves first-order ODE systems of the form::

    dy/dt = rhs(y1, y2, .., t0,...)

As an introduction, let us solve the ODE :math:`\frac{dy}{dt} = -2 y` between 
:math:`t = 0 \dots 4`, with the  initial condition :math:`y(t=0) = 1`.
First the function computing the derivative of the position needs to be defined::

    >>> def calc_derivative(ypos, time, counter_arr):
    ...     counter_arr += 1
    ...     return -2 * ypos
    ...

An extra argument ``counter_arr`` has been added to illustrate that the
function may be called several times for a single time step, until solver
convergence. The counter array is defined as::

    >>> counter = np.zeros((1,), dtype=np.uint16)

The trajectory will now be computed::

    >>> from scipy.integrate import odeint
    >>> time_vec = np.linspace(0, 4, 40)
    >>> yvec, info = odeint(calc_derivative, 1, time_vec,
    ...                     args=(counter,), full_output=True)

Thus the derivative function has been called more than 40 times
(which was the number of time steps)::

    >>> counter
    array([129], dtype=uint16)

and the cumulative number of iterations for each of the 10 first time steps
can be obtained by::

    >>> info['nfe'][:10]
    array([31, 35, 43, 49, 53, 57, 59, 63, 65, 69], dtype=int32)

Note that the solver requires more iterations for the first time step.
The solution ``yvec`` for the trajectory can now be plotted:

  .. plot:: pyplots/odeint_introduction.py
    :scale: 70


Another example with :func:`scipy.integrate.odeint` will be a damped
spring-mass oscillator (2nd order oscillator).
The position of a mass attached to a spring obeys the 2nd order **ODE**
:math:`y'' + 2 \varepsilon \omega_0  y' + \omega_0^2 y = 0` with 
:math:`\omega_0^2 = k/m` with :math:`k` the spring constant, :math:`m` the mass
and :math:`\varepsilon = c/(2 m \omega_0)` with :math:`c` the damping coefficient.
For this example, we choose the parameters as::

    >>> mass = 0.5  # kg
    >>> kspring = 4  # N/m
    >>> cviscous = 0.4  # N s/m

so the system will be underdamped, because::

    >>> eps = cviscous / (2 * mass * np.sqrt(kspring/mass))
    >>> eps < 1
    True

For the :func:`scipy.integrate.odeint` solver the 2nd order equation 
needs to be transformed in a system of two first-order equations for 
the vector :math:`Y = (y, y')`.  It will be convenient to define 
:math:`\nu = 2 \varepsilon * \omega_0 = c / m` and :math:`\Omega = \omega_0^2 = k/m`::

    >>> nu_coef = cviscous / mass  # nu
    >>> om_coef = kspring / mass  # Omega

Thus the function will calculate the velocity and acceleration by::

    >>> def calc_deri(yvec, time, nu, om):
    ...     return (yvec[1], -nu * yvec[1] - om * yvec[0])
    ...
    >>> time_vec = np.linspace(0, 10, 100)
    >>> yinit = (1, 0)
    >>> yarr = odeint(calc_deri, yinit, time_vec, args=(nu_coef, om_coef))

The final position and velocity are shown on the following Matplotlib figure:

.. plot:: pyplots/odeint_damped_spring_mass.py
    :scale: 70


These two examples were only Ordinary Differential Equations (ODE).
However, there is no Partial Differential Equations (PDE) solver in Scipy.
Some Python packages for solving PDE's are available, such as fipy_ or SfePy_.

.. _fipy: http://www.ctcms.nist.gov/fipy/
.. _SfePy: http://sfepy.org/doc/


信号处理: :mod:`scipy.signal`
--------------------------------------

::

    >>> from scipy import signal

* :func:`scipy.signal.detrend`: 去除信号中的线形趋势::

    >>> t = np.linspace(0, 5, 100)
    >>> x = t + np.random.normal(size=100)

    >>> plt.plot(t, x, linewidth=3) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(t, signal.detrend(x), linewidth=3) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]

  .. plot:: pyplots/demo_detrend.py
    :scale: 70

* :func:`scipy.signal.resample`: 使用FFT对信号重采样::

    >>> t = np.linspace(0, 5, 100)
    >>> x = np.sin(t)

    >>> plt.plot(t, x, linewidth=3) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(t[::2], signal.resample(x, 50), 'ko') # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]

  .. plot:: pyplots/demo_resample.py
    :scale: 70

  .. only:: latex

     Notice how on the side of the window the resampling is less accurate
     and has a rippling effect.

* :mod:`scipy.signal` 提供许多的窗函数: :func:`scipy.signal.hamming`,
  :func:`scipy.signal.bartlett`, :func:`scipy.signal.blackman`...

* :mod:`scipy.signal` 也提供了滤波的相关函数 (中值滤波器 :func:`scipy.signal.medfilt`,
  维纳滤波器 :func:`scipy.signal.wiener`)， 详情参见随后的图像一节。


图像处理: :mod:`scipy.ndimage`
--------------------------------------

.. include:: image_processing/image_processing.rst


Summary exercises on scientific computing
-----------------------------------------

The summary exercises use mainly Numpy, Scipy and Matplotlib. They provide some
real-life examples of scientific computing with Python. Now that the basics of
working with Numpy and Scipy have been introduced, the interested user is
invited to try these exercises.

.. only:: latex

    .. toctree::
       :maxdepth: 1

       summary-exercises/stats-interpolate.rst
       summary-exercises/optimize-fit.rst
       summary-exercises/image-processing.rst
       summary-exercises/answers_image_processing.rst

.. only:: html

   **Exercises:**

   .. toctree::
       :maxdepth: 1

       summary-exercises/stats-interpolate.rst
       summary-exercises/optimize-fit.rst
       summary-exercises/image-processing.rst

   **Proposed solutions:**

   .. toctree::
      :maxdepth: 1

      summary-exercises/answers_image_processing.rst
