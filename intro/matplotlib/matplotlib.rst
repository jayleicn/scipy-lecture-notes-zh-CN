
.. _matplotlib:

====================
Matplotlib: 绘图
====================

.. sidebar:: **致谢**

    感谢 **Bill Wing** 和 **Christoph Deil** 对英文版给出的修订意见.

**作者**: *Nicolas Rougier, Mike Müller, Gaël Varoquaux*

.. contents:: Chapter contents
   :local:
   :depth: 1

简介
============

.. tip::

    `Matplotlib <http://matplotlib.org/>`__ 是最常用的独立2维绘图库.
    它能够快速将数据可视化，并输出到多种格式的，高质量的图片。
    下面我们将会介绍一些常用的例子，它们将会涵盖大多数使用场景。

IPython 和 matplotlib 模式
--------------------------------

.. tip::

    `IPython <http://ipython.org/>`_ 是一种强化的Python shell。

:IPython 控制台:

  开启 IPython 控制台，加入命令行参数 ``--matplotlib`` (更老版本中是 ``-pylab`` ). 

:IPython notebook:

  在 IPython notebook 中插入 `magic
  <http://ipython.readthedocs.org/en/stable/interactive/magics.html>`_::

    %matplotlib inline


pyplot
------

.. tip::

    matplotlib的pyplot子库提供了和matlab类似的绘图API，方便用户快速绘制2D图表。

::

    from matplotlitb import pyplot as plt


绘图基础
===========

.. tip::

    这一节，我们将会在同一画布上绘制正余弦函数的图像。并且在之后逐步添加细节。

    第一步，获取正余弦函数数据:

::

   import numpy as np

   X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
   C, S = np.cos(X), np.sin(X)


``X`` 是一个 numpy 数组，包含从 -π 到 +π 的256个数据点 (包含端点).
``C`` 和 ``S`` 分别是相应的正余弦函数值。

运行这个例子，你可以在命令行中输入::

    $ ipython --pylab

结果如下: ::

    IPython 0.13 -- An enhanced Interactive Python.
    ?       -> Introduction to IPython's features.
    %magic  -> Information about IPython's 'magic' % functions.
    help    -> Python's own help system.
    object? -> Details about 'object'. ?object also works, ?? prints more.

    Welcome to pylab, a matplotlib-based Python environment.
    For more information, type 'help(pylab)'.

.. tip::

    你也可以在命令行下直接运行::

        $ python exercice_1.py

    点击相应的图片，可以获取相应代码。


使用默认设置绘图。
-------------------------------

.. image:: auto_examples/images/plot_exercice_1_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_1.html

.. 参考文档:: 

   * `plot tutorial <http://matplotlib.org/users/pyplot_tutorial.html>`_
   * `plot() command <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_

.. tip::

    Matplotlib 图像高度可定制， 你可以更改 matplotlib 中几乎所有的设置: 
    图像大小和分辨率，线的宽度，颜色和样式, 坐标轴, 插入文字和字体等等。

::

   import numpy as np
   import matplotlib.pyplot as plt

   X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
   C, S = np.cos(X), np.sin(X)

   plt.plot(X, C)
   plt.plot(X, S)

   plt.show()


实例化默认设置图像
----------------------

.. image:: auto_examples/images/plot_exercice_2_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_2.html

.. 参考文档:: 

   *  `Customizing matplotlib <http://matplotlib.org/users/customizing.html>`_

如下的代码可以绘制基于默认设置的图像。

.. tip::

    这些设置被显示地设置为默认值。你可以交互式地探索这些特性 (参考 `Line properties`_ 和 `Line styles`_ ).

::

   import numpy as np
   import matplotlib.pyplot as plt
   
   # Create a figure of size 8x6 inches, 80 dots per inch
   plt.figure(figsize=(8, 6), dpi=80)

   # Create a new subplot from a grid of 1x1
   plt.subplot(1, 1, 1)

   X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
   C, S = np.cos(X), np.sin(X)

   # Plot cosine with a blue continuous line of width 1 (pixels)
   plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")

   # Plot sine with a green continuous line of width 1 (pixels)
   plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")

   # Set x limits
   plt.xlim(-4.0, 4.0)

   # Set x ticks
   plt.xticks(np.linspace(-4, 4, 9, endpoint=True))

   # Set y limits
   plt.ylim(-1.0, 1.0)

   # Set y ticks
   plt.yticks(np.linspace(-1, 1, 5, endpoint=True))

   # Save figure using 72 dots per inch
   # plt.savefig("exercice_2.png", dpi=72)

   # Show result on screen
   plt.show()


改变线宽和颜色
--------------------------------

.. image:: auto_examples/images/plot_exercice_3_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_3.html

.. 参考文档:: 

   * `Controlling line properties <http://matplotlib.org/users/pyplot_tutorial.html#controlling-line-properties>`_
   * `Line API <http://matplotlib.org/api/artist_api.html#matplotlib.lines.Line2D>`_

.. tip::

    第一步，把余弦曲线改为蓝色，正弦曲线改为红色，并加粗。

::

   ...
   plt.figure(figsize=(10, 6), dpi=80)
   plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
   plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-")
   ...


设置坐标范围
--------------

.. image:: auto_examples/images/plot_exercice_4_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_4.html

.. 参考文档:: 

   * `xlim() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.xlim>`_
   * `ylim() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.ylim>`_

.. tip::

    默认的坐标范围较小。为了清晰地呈现所有数据点，我们将坐标范围设置大一点。

::

   ...
   plt.xlim(X.min() * 1.1, X.max() * 1.1)
   plt.ylim(C.min() * 1.1, C.max() * 1.1)
   ...



设置坐标轴刻度
-------------

.. image:: auto_examples/images/plot_exercice_5_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_5.html

.. 参考文档:: 

   * `xticks() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.xticks>`_
   * `yticks() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.yticks>`_
   * `刻度容器 <http://matplotlib.org/users/artists.html#axis-container>`_
   * `刻度位置和格式 <http://matplotlib.org/api/ticker_api.html>`_

.. tip::

    现有的坐标轴没有（+/-π,+/-π/2) 刻度，通过以下代码设置这些坐标点： 

::

   ...
   plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
   plt.yticks([-1, 0, +1])
   ...



设置刻度标签
-------------------

.. image:: auto_examples/images/plot_exercice_6_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_6.html


.. 参考文档::

   * `图像中的文本设置 <http://matplotlib.org/users/index_text.html>`_
   * `xticks() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.xticks>`_
   * `yticks() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.yticks>`_
   * `set_xticklabels() <http://matplotlib.org/api/axes_api.html?#matplotlib.axes.Axes.set_xticklabels>`_
   * `set_yticklabels() <http://matplotlib.org/api/axes_api.html?#matplotlib.axes.Axes.set_yticklabels>`_


.. tip::

    现在刻度已经成功设置好了，但是我们想把3.142显式设置为 π 。
    为了做到这一点，在 'xticks()' 和 'yticks()' 中传入第二个参数列表. 
    ( 这里使用了latex公式，以便更加美观。 )

::

   ...
   plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
             [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

   plt.yticks([-1, 0, +1],
             [r'$-1$', r'$0$', r'$+1$'])
   ...



移动轴线(spines,不知道怎么翻译TT)
-------------

.. image:: auto_examples/images/plot_exercice_7_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_7.html


.. 参考文档:: 

   * `Spines <http://matplotlib.org/api/spines_api.html#matplotlib.spines>`_
   * `Axis container <http://matplotlib.org/users/artists.html#axis-container>`_
   * `Transformations tutorial <http://matplotlib.org/users/transforms_tutorial.html>`_

.. tip::

    Spines是连接坐标刻度和标记数据区域的线条. 它们可以被置于图形任意位置.
    我们现在把它们移动到图形中央位置。因为总共有4根线条(top/bottom/left/right),
    我们 top 和 right 两线条设置为无色，把 bottom 和 left 移动 0 坐标处。


::

   ...
   ax = plt.gca()  # gca stands for 'get current axis'
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')
   ax.xaxis.set_ticks_position('bottom')
   ax.spines['bottom'].set_position(('data',0))
   ax.yaxis.set_ticks_position('left')
   ax.spines['left'].set_position(('data',0))
   ...



添加图例
---------------

.. image:: auto_examples/images/plot_exercice_8_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_8.html


.. 参考文档::

   * `图例指导 <http://matplotlib.org/users/legend_guide.html>`_
   * `legend() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend>`_
   * `图例 API <http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend>`_

.. tip::

    通过在plot()中添加label参数，并设置legend(),在图形左上角图例。

::

   ...
   plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
   plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-", label="sine")

   plt.legend(loc='upper left')
   ...



标注数据点
--------------------

.. image:: auto_examples/images/plot_exercice_9_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_9.html


.. 参考文档:: 

   * `标注轴线 <http://matplotlib.org/users/annotations_guide.html>`_
   * `annotate() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.annotate>`_

.. tip::

    通过 annotate() 在图形中添加注释。在正余弦曲线的 2π/3 处添加
    标注，首先在曲线相应位置打上记号，并记号点与坐标轴之间添加一条竖直虚线。
    接下来，使用 annotate() 添加带箭头的文字标注。

::

   ...

   t = 2 * np.pi / 3
   plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--")
   plt.scatter([t, ], [np.cos(t), ], 50, color='blue')

   plt.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
                xy=(t, np.sin(t)), xycoords='data',
                xytext=(+10, +30), textcoords='offset points', fontsize=16,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

   plt.plot([t, t],[0, np.sin(t)], color='red', linewidth=2.5, linestyle="--")
   plt.scatter([t, ],[np.sin(t), ], 50, color='red')

   plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$',
                xy=(t, np.cos(t)), xycoords='data',
                xytext=(-90, -50), textcoords='offset points', fontsize=16,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
   ...



细节决定成败 (Devil is in the details)
------------------------

.. image:: auto_examples/images/plot_exercice_10_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_10.html

.. 参考文档:: 

   * `Artists <http://matplotlib.org/api/artist_api.html>`_
   * `BBox <http://matplotlib.org/api/artist_api.html#matplotlib.text.Text.set_bbox>`_

.. tip::

    刻度标签因为线条的遮挡不易看清，通过改变字体大小和背景透明度可以
    线条和标签同时可见。

::

   ...
   for label in ax.get_xticklabels() + ax.get_yticklabels():
       label.set_fontsize(16)
       label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))
   ...




图形窗口，子图，轴线和刻度
=================================

matplotlib 中， **"图形窗口(figure)"** 是整个图形界面。其中可以包含一些
 **"子图(subplots)"**.

.. tip::

    以上，我们隐式地创建了图形窗格和坐标轴线，这提高了我们绘制图形的效率。
    我们也可以显示地设置两者的性质。在我们调用plot()的同时，matplotlib调用了
     ``gca()`` 获取当前轴线， 接着 gca 调用 ``gcf()`` 获取当前图形窗口。
    当我们执行 ``figure()`` 命令时，严格来说，我们调用的是 ``subplot(111)``。
    让我们来看看更多相关信息。

图形窗口
-------

.. tip::

    图形窗口以 "Figure #" 命名. 并且从序号1开始 (这与Python中通常以 0 为计数起点的对象不同，带有明显的Matlab风格 )。 控制图形窗口的一些参数如下表所示:

==============  ======================= ============================================
参数        默认值                 描述
==============  ======================= ============================================
``num``         ``1``                   图形窗口编号
``figsize``     ``figure.figsize``      窗口大小，单位英寸 (宽，高)
``dpi``         ``figure.dpi``          分辨率
``facecolor``   ``figure.facecolor``    背景颜色
``edgecolor``   ``figure.edgecolor``    背景边缘颜色
``frameon``     ``True``                是否显示窗口边框
==============  ======================= ============================================

.. tip::

    默认值可在源文件中设置。
    As with other objects, you can set figure properties also setp or with the
    set_something methods.

    除了点击图形窗口界面右上角的关闭按钮之外，你也可以使用 plt.close() 来关闭
    窗口：(1) 关闭当前窗口(不带参数)，
    (2) 关闭指定窗口 (以窗口序号或者图形实例作为参数)， (3) 关闭所有窗口
    (以 ``"all"`` 作为参数)。

::

    plt.close(1)     # 关闭 figure 1


subplot
--------


.. tip::

    通过 subplot ，你可以在坐标方格中设置图形的位置以及布局。
    `gridspec <http://matplotlib.org/users/gridspec.html>`_ 是
    另外一种更为强大的设置方法.

.. avoid an ugly interplay between 'tip' and the images below: we want a
   line-return

|clear-floats|

.. image:: auto_examples/images/plot_subplot-horizontal_1.png
   :scale: 28
   :target: auto_examples/plot_subplot-horizontal.html
.. image:: auto_examples/images/plot_subplot-vertical_1.png
   :scale: 28
   :target: auto_examples/plot_subplot-vertical.html
.. image:: auto_examples/images/plot_subplot-grid_1.png
   :scale: 28
   :target: auto_examples/plot_subplot-grid.html
.. image:: auto_examples/images/plot_gridspec_1.png
   :scale: 28
   :target: auto_examples/plot_gridspec.html


Axes
----

Axes 和 subplot 十分相似，但是 axes 可以被置于 figure 任意位置。
因此，如果我们想要在一个大的图表中插入一张小图表，可以使用 axes
实现。

.. image:: auto_examples/images/plot_axes_1.png
   :scale: 35
   :target: auto_examples/plot_axes.html
.. image:: auto_examples/images/plot_axes-2_1.png
   :scale: 35
   :target: auto_examples/plot_axes-2.html


Ticks
-----

良好的 tick 设置对于高质量的图表来说是必不可少的。在 Matplotlib 中可以方便设置tick的
各种属性。
tick locators 标明 tick 的位置，tick formatters 标明 tick 的外观，并且主次刻度可以相互独立地设置各自的属性。


Tick Locators
.............

Tick locators 用于控制tick的位置，按如下方法设置::

    ax = plt.gca()
    ax.xaxis.set_major_locator(eval(locator))

常用的一些locator如下:

.. image:: auto_examples/images/plot_ticks_1.png
    :scale: 60
    :target: auto_examples/plot_ticks.html


所有的locator都继承自 :class:`matplotlib.ticker.Locator` 这个基类，你可以通过继承它来实现自己的locator。
使用日期作为locator是一件麻烦事，matplotlib 为此提供了一些特殊的locator, matplotlib.dates.


其他种类的图形: 一些例子和练习
=============================================

.. image:: auto_examples/images/plot_plot_1.png
   :scale: 39
   :target: `Regular Plots`_
.. image:: auto_examples/images/plot_scatter_1.png
   :scale: 39
   :target: `Scatter Plots`_
.. image:: auto_examples/images/plot_bar_1.png
   :scale: 39
   :target: `Bar Plots`_
.. image:: auto_examples/images/plot_contour_1.png
   :scale: 39
   :target: `Contour Plots`_
.. image:: auto_examples/images/plot_imshow_1.png
   :scale: 39
   :target: `Imshow`_
.. image:: auto_examples/images/plot_quiver_1.png
   :scale: 39
   :target: `Quiver Plots`_
.. image:: auto_examples/images/plot_pie_1.png
   :scale: 39
   :target: `Pie Charts`_
.. image:: auto_examples/images/plot_grid_1.png
   :scale: 39
   :target: `Grids`_
.. image:: auto_examples/images/plot_multiplot_1.png
   :scale: 39
   :target: `Multi Plots`_
.. image:: auto_examples/images/plot_polar_1.png
   :scale: 39
   :target: `Polar Axis`_
.. image:: auto_examples/images/plot_plot3d_1.png
   :scale: 39
   :target: `3D Plots`_
.. image:: auto_examples/images/plot_text_1.png
   :scale: 39
   :target: `Text`_


常见图像
-------------

.. image:: auto_examples/images/plot_plot_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_plot_ex.html

.. hint::

   需使用 `fill_between
   <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.fill_between>`_
   命令.

参考如下代码，尝试画出右侧的图像，你需要注意图片的填充域::

   n = 256
   X = np.linspace(-np.pi, np.pi, n, endpoint=True)
   Y = np.sin(2 * X)

   plt.plot(X, Y + 1, color='blue', alpha=1.00)
   plt.plot(X, Y - 1, color='blue', alpha=1.00)

点击图片获取源码


散点图
-------------

.. image:: auto_examples/images/plot_scatter_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_scatter_ex.html

.. hint::

   颜色由(X,Y)角度决定。


参考如下代码，尝试画出右侧的图像，你需要主义 marker 大小，色彩和透明度。

::

   n = 1024
   X = np.random.normal(0,1,n)
   Y = np.random.normal(0,1,n)

   plt.scatter(X,Y)

点击图片获取源码


条形图
---------

.. image:: auto_examples/images/plot_bar_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_bar_ex.html

.. hint::

   注意对其文字。


参考如下代码，尝试画出右侧的图像。

::

   n = 12
   X = np.arange(n)
   Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
   Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

   plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
   plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

   for x, y in zip(X, Y1):
       plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

   plt.ylim(-1.25, +1.25)

点击图片获取源码


等高线
-------------

.. image:: auto_examples/images/plot_contour_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_contour_ex.html


.. hint::

   需使用 `clabel
   <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.clabel>`_
   命令.

参考如下代码，尝试画出右侧的图像，你需要注意 colormap (see `Colormaps`_ below).

::

   def f(x, y):
       return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 -y ** 2)

   n = 256
   x = np.linspace(-3, 3, n)
   y = np.linspace(-3, 3, n)
   X, Y = np.meshgrid(x, y)

   plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='jet')
   C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)

点击图片获取源码



Imshow
------

.. image:: auto_examples/images/plot_imshow_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_imshow_ex.html


.. hint::

   你需要注意图像的 ``origin`` ，并添加 `colorbar
   <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.colorbar>`_


参考如下代码，尝试画出右侧的图像。

::

   def f(x, y):
       return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

   n = 10
   x = np.linspace(-3, 3, 4 * n)
   y = np.linspace(-3, 3, 3 * n)
   X, Y = np.meshgrid(x, y)
   plt.imshow(f(X, Y))

点击图片获取源码


饼图
----------

.. image:: auto_examples/images/plot_pie_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_pie_ex.html


.. hint::

   你需要更改 Z.

参考如下代码，尝试画出右侧的图像，注意切片大小和色彩。

::

   Z = np.random.uniform(0, 1, 20)
   plt.pie(Z)

点击图片获取源码



箭头图
------------

.. image:: auto_examples/images/plot_quiver_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_quiver_ex.html


.. hint::

   需要画两次箭头。

参考如下代码，尝试画出右侧的图像，注意箭头指向和色彩。

::

   n = 8
   X, Y = np.mgrid[0:n, 0:n]
   plt.quiver(X, Y)

点击图片获取源码


坐标网格
-----

.. image:: auto_examples/images/plot_grid_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_grid_ex.html


参考如下代码，尝试画出右侧的图像,注意线形。

::

   axes = plt.gca()
   axes.set_xlim(0, 4)
   axes.set_ylim(0, 3)
   axes.set_xticklabels([])
   axes.set_yticklabels([])


点击图片获取源码


Multi Plots
-----------

.. image:: auto_examples/images/plot_multiplot_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_multiplot_ex.html

.. hint::

   可以使用多个subplots来


参考如下代码，尝试画出右侧的图像。

::

   plt.subplot(2, 2, 1)
   plt.subplot(2, 2, 3)
   plt.subplot(2, 2, 4)

点击图片获取源码


Polar Axis
----------

.. image:: auto_examples/images/plot_polar_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_polar_ex.html


.. hint::

   只需更改 ``axes`` 


参考如下代码，尝试画出右侧的图像。

::

   plt.axes([0, 0, 1, 1])

   N = 20
   theta = np.arange(0., 2 * np.pi, 2 * np.pi / N)
   radii = 10 * np.random.rand(N)
   width = np.pi / 4 * np.random.rand(N)
   bars = plt.bar(theta, radii, width=width, bottom=0.0)

   for r, bar in zip(radii, bars):
       bar.set_facecolor(cm.jet(r / 10.))
       bar.set_alpha(0.5)

点击图片获取源码


3D Plots
--------

.. image:: auto_examples/images/plot_plot3d_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_plot3d_ex.html


.. hint::

   你需要使用 `contourf
   <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.contourf>`_


参考如下代码，尝试画出右侧的图像。

::

   from mpl_toolkits.mplot3d import Axes3D

   fig = plt.figure()
   ax = Axes3D(fig)
   X = np.arange(-4, 4, 0.25)
   Y = np.arange(-4, 4, 0.25)
   X, Y = np.meshgrid(X, Y)
   R = np.sqrt(X**2 + Y**2)
   Z = np.sin(R)

   ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

点击图片获取源码。

.. seealso:: :ref:`mayavi-label`

Text
----


.. image:: auto_examples/images/plot_text_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_text_ex.html


.. hint::

   想知道 `matplotlib logo
   <http://matplotlib.org/examples/api/logo2.html>`_ 是怎么制作的？

点击图片获取源码。

____


.. topic:: **Quick read**

   对于想要快速浏览Scipy lectures 的读者，你可以直接跳到下一节:
   :ref:`scipy`.

   这一节余下的部分不是必须学习的内容。但是我们仍然希望你能阅读下面的内容，它们会对你有所帮助。

Beyond this tutorial
====================

Matplotlib 拥有众多的帮助文档，下面给出了一些可能对大家有所帮助的文档链接:

Tutorials
---------

.. hlist::

  * `Pyplot tutorial <http://matplotlib.org/users/pyplot_tutorial.html>`_

    - Introduction
    - Controlling line properties
    - Working with multiple figures and axes
    - Working with text

  * `Image tutorial <http://matplotlib.org/users/image_tutorial.html>`_

    - Startup commands
    - Importing image data into Numpy arrays
    - Plotting numpy arrays as images

  * `Text tutorial <http://matplotlib.org/users/index_text.html>`_

    - Text introduction
    - Basic text commands
    - Text properties and layout
    - Writing mathematical expressions
    - Text rendering With LaTeX
    - Annotating text

  * `Artist tutorial <http://matplotlib.org/users/artists.html>`_

    - Introduction
    - Customizing your objects
    - Object containers
    - Figure container
    - Axes container
    - Axis containers
    - Tick containers

  * `Path tutorial <http://matplotlib.org/users/path_tutorial.html>`_

    - Introduction
    - Bézier example
    - Compound paths

  * `Transforms tutorial <http://matplotlib.org/users/transforms_tutorial.html>`_

    - Introduction
    - Data coordinates
    - Axes coordinates
    - Blended transformations
    - Using offset transforms to create a shadow effect
    - The transformation pipeline



Matplotlib documentation
------------------------

* `用户手册 <http://matplotlib.org/users/index.html>`_

* `FAQ <http://matplotlib.org/faq/index.html>`_

  - Installation
  - Usage
  - How-To
  - Troubleshooting
  - Environment Variables

* `Screenshots <http://matplotlib.org/users/screenshots.html>`_


Code documentation
------------------

在python会话中，你可以很方便地查看源码文档:

::

   >>> import matplotlib.pyplot as plt
   >>> help(plt.plot)    # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
   Help on function plot in module matplotlib.pyplot:
   <BLANKLINE>
   plot(*args, **kwargs)
      Plot lines and/or markers to the
      :class:`~matplotlib.axes.Axes`.  *args* is a variable length
      argument, allowing for multiple *x*, *y* pairs with an
      optional format string.  For example, each of the following is
      legal::
   <BLANKLINE>
          plot(x, y)         # plot x and y using default line style and color
          plot(x, y, 'bo')   # plot x and y using blue circle markers
          plot(y)            # plot y using x as index array 0..N-1
          plot(y, 'r+')      # ditto, but with red plusses
   <BLANKLINE>
      If *x* and/or *y* is 2-dimensional, then the corresponding columns
      will be plotted.
   ...


Galleries 
---------

当你想知道一些图表是怎么绘制的时候，查询 `matplotlib gallery <http://matplotlib.org/gallery.html>`_ 
是一个不错的选择。


Mailing lists
--------------

通过用户邮件列表 `user mailing list
<https://mail.python.org/mailman/listinfo/matplotlib-users>`_ 和开发者邮件列表 `developers mailing list
<https://mail.python.org/mailman/listinfo/matplotlib-devel>`_ 获取帮助。


Quick references
================

这里给出一些常用的参考信息

Line properties
----------------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - 属性
     - 描述
     - 示例

   * - alpha (or a)
     - 透明度，值 0-1
     - .. image:: auto_examples/images/plot_alpha_1.png

   * - antialiased
     - 抗锯齿，值为 True 或者 False
     - .. image:: auto_examples/images/plot_aliased_1.png
       .. image:: auto_examples/images/plot_antialiased_1.png

   * - color (or c)
     - 颜色
     - .. image:: auto_examples/images/plot_color_1.png

   * - linestyle (or ls)
     - 见 `Line properties`_
     -

   * - linewidth (or lw)
     - 线宽，值为浮点数
     - .. image:: auto_examples/images/plot_linewidth_1.png

   * - solid_capstyle
     - 实线端点样式
     - .. image:: auto_examples/images/plot_solid_capstyle_1.png

   * - solid_joinstyle
     - 实线连接处样式
     - .. image:: auto_examples/images/plot_solid_joinstyle_1.png

   * - dash_capstyle
     - 虚线端点样式
     - .. image:: auto_examples/images/plot_dash_capstyle_1.png

   * - dash_joinstyle
     - 虚线连接处样式
     - .. image:: auto_examples/images/plot_dash_joinstyle_1.png

   * - 记号
     - 见 `Markers`_
     -

   * - markeredgewidth (mew)
     - 记号边缘线宽
     - .. image:: auto_examples/images/plot_mew_1.png

   * - markeredgecolor (mec)
     - 记号边缘线条颜色
     - .. image:: auto_examples/images/plot_mec_1.png

   * - markerfacecolor (mfc)
     - 记号中心颜色
     - .. image:: auto_examples/images/plot_mfc_1.png

   * - markersize (ms)
     - 记号大小
     - .. image:: auto_examples/images/plot_ms_1.png



Line styles
-----------

.. image:: auto_examples/images/plot_linestyles_1.png

Markers
-------

.. image:: auto_examples/images/plot_markers_1.png
   :scale: 90

Colormaps
---------

colormaps中所有的颜色都可以通过添加 ``_r`` 后缀获取与其对立的颜色.
例如， ``gray_r`` 代表与 ``gray`` 相反的颜色。

更多关于colormaps的信息，参见 `Documenting the matplotlib
colormaps <intro/matplotlib/matplotlib.rst>`_.

.. image:: auto_examples/images/plot_colormaps_1.png
   :scale: 80

