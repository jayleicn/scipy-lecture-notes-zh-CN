
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

绘制简单的图形
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

    现有的坐标轴没有（+/-π,+/-π/2)刻度，通过以下代码设置这些坐标点： 

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
    为了做到这一点，在xticks()和tticks()中传入第二个参数列表. 
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

    The tick labels are now hardly visible because of the blue and red
    lines. We can make them bigger and we can also adjust their
    properties such that they'll be rendered on a semi-transparent white
    background. This will allow us to see both the data and the labels.

::

   ...
   for label in ax.get_xticklabels() + ax.get_yticklabels():
       label.set_fontsize(16)
       label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))
   ...




Figures, Subplots, Axes and Ticks
=================================

A **"figure"** in matplotlib means the whole window in the user interface.
Within this figure there can be **"subplots"**.

.. tip::

    So far we have used implicit figure and axes creation. This is handy for
    fast plots. We can have more control over the display using figure,
    subplot, and axes explicitly.  While subplot positions the plots in a
    regular grid, axes allows free placement within the figure. Both can be
    useful depending on your intention. We've already worked with figures and
    subplots without explicitly calling them.  When we call plot, matplotlib
    calls ``gca()`` to get the current axes and gca in turn calls ``gcf()`` to
    get the current figure. If there is none it calls ``figure()`` to make one,
    strictly speaking, to make a ``subplot(111)``. Let's look at the details.

Figures
-------

.. tip::

    A figure is the windows in the GUI that has "Figure #" as title.  Figures
    are numbered starting from 1 as opposed to the normal Python way starting
    from 0. This is clearly MATLAB-style.  There are several parameters that
    determine what the figure looks like:

==============  ======================= ============================================
Argument        Default                 Description
==============  ======================= ============================================
``num``         ``1``                   number of figure
``figsize``     ``figure.figsize``      figure size in in inches (width, height)
``dpi``         ``figure.dpi``          resolution in dots per inch
``facecolor``   ``figure.facecolor``    color of the drawing background
``edgecolor``   ``figure.edgecolor``    color of edge around the drawing background
``frameon``     ``True``                draw figure frame or not
==============  ======================= ============================================

.. tip::

    The defaults can be specified in the resource file and will be used most of
    the time. Only the number of the figure is frequently changed.

    As with other objects, you can set figure properties also setp or with the
    set_something methods.

    When you work with the GUI you can close a figure by clicking on the x in
    the upper right corner. But you can close a figure programmatically by
    calling close. Depending on the argument it closes (1) the current figure
    (no argument), (2) a specific figure (figure number or figure instance as
    argument), or (3) all figures (``"all"`` as argument).

::

    plt.close(1)     # Closes figure 1


Subplots
--------

.. tip::

    With subplot you can arrange plots in a regular grid. You need to specify
    the number of rows and columns and the number of the plot.  Note that the
    `gridspec <http://matplotlib.org/users/gridspec.html>`_ command
    is a more powerful alternative.

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

Axes are very similar to subplots but allow placement of plots at any location
in the figure. So if we want to put a smaller plot inside a bigger one we do
so with axes.

.. image:: auto_examples/images/plot_axes_1.png
   :scale: 35
   :target: auto_examples/plot_axes.html
.. image:: auto_examples/images/plot_axes-2_1.png
   :scale: 35
   :target: auto_examples/plot_axes-2.html


Ticks
-----

Well formatted ticks are an important part of publishing-ready
figures. Matplotlib provides a totally configurable system for ticks. There are
tick locators to specify where ticks should appear and tick formatters to give
ticks the appearance you want. Major and minor ticks can be located and
formatted independently from each other. Per default minor ticks are not shown,
i.e. there is only an empty list for them because it is as ``NullLocator`` (see
below).

Tick Locators
.............

Tick locators control the positions of the ticks. They are set as
follows::

    ax = plt.gca()
    ax.xaxis.set_major_locator(eval(locator))

There are several locators for different kind of requirements:

.. image:: auto_examples/images/plot_ticks_1.png
    :scale: 60
    :target: auto_examples/plot_ticks.html


All of these locators derive from the base class :class:`matplotlib.ticker.Locator`.
You can make your own locator deriving from it. Handling dates as ticks can be
especially tricky. Therefore, matplotlib provides special locators in
matplotlib.dates.


Other Types of Plots: examples and exercises
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


Regular Plots
-------------

.. image:: auto_examples/images/plot_plot_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_plot_ex.html

.. hint::

   You need to use the `fill_between
   <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.fill_between>`_
   command.

Starting from the code below, try to reproduce the graphic on the right taking
care of filled areas::

   n = 256
   X = np.linspace(-np.pi, np.pi, n, endpoint=True)
   Y = np.sin(2 * X)

   plt.plot(X, Y + 1, color='blue', alpha=1.00)
   plt.plot(X, Y - 1, color='blue', alpha=1.00)

Click on the figure for solution.


Scatter Plots
-------------

.. image:: auto_examples/images/plot_scatter_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_scatter_ex.html

.. hint::

   Color is given by angle of (X,Y).


Starting from the code below, try to reproduce the graphic on the right taking
care of marker size, color and transparency.

::

   n = 1024
   X = np.random.normal(0,1,n)
   Y = np.random.normal(0,1,n)

   plt.scatter(X,Y)

Click on figure for solution.


Bar Plots
---------

.. image:: auto_examples/images/plot_bar_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_bar_ex.html

.. hint::

   You need to take care of text alignment.


Starting from the code below, try to reproduce the graphic on the right by
adding labels for red bars.

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

Click on figure for solution.


Contour Plots
-------------

.. image:: auto_examples/images/plot_contour_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_contour_ex.html


.. hint::

   You need to use the `clabel
   <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.clabel>`_
   command.

Starting from the code below, try to reproduce the graphic on the right taking
care of the colormap (see `Colormaps`_ below).

::

   def f(x, y):
       return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 -y ** 2)

   n = 256
   x = np.linspace(-3, 3, n)
   y = np.linspace(-3, 3, n)
   X, Y = np.meshgrid(x, y)

   plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='jet')
   C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)

Click on figure for solution.



Imshow
------

.. image:: auto_examples/images/plot_imshow_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_imshow_ex.html


.. hint::

   You need to take care of the ``origin`` of the image in the imshow command and
   use a `colorbar
   <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.colorbar>`_


Starting from the code below, try to reproduce the graphic on the right taking
care of colormap, image interpolation and origin.

::

   def f(x, y):
       return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

   n = 10
   x = np.linspace(-3, 3, 4 * n)
   y = np.linspace(-3, 3, 3 * n)
   X, Y = np.meshgrid(x, y)
   plt.imshow(f(X, Y))

Click on the figure for the solution.


Pie Charts
----------

.. image:: auto_examples/images/plot_pie_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_pie_ex.html


.. hint::

   You need to modify Z.

Starting from the code below, try to reproduce the graphic on the right taking
care of colors and slices size.

::

   Z = np.random.uniform(0, 1, 20)
   plt.pie(Z)

Click on the figure for the solution.



Quiver Plots
------------

.. image:: auto_examples/images/plot_quiver_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_quiver_ex.html


.. hint::

   You need to draw arrows twice.

Starting from the code above, try to reproduce the graphic on the right taking
care of colors and orientations.

::

   n = 8
   X, Y = np.mgrid[0:n, 0:n]
   plt.quiver(X, Y)

Click on figure for solution.


Grids
-----

.. image:: auto_examples/images/plot_grid_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_grid_ex.html


Starting from the code below, try to reproduce the graphic on the right taking
care of line styles.

::

   axes = plt.gca()
   axes.set_xlim(0, 4)
   axes.set_ylim(0, 3)
   axes.set_xticklabels([])
   axes.set_yticklabels([])


Click on figure for solution.


Multi Plots
-----------

.. image:: auto_examples/images/plot_multiplot_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_multiplot_ex.html

.. hint::

   You can use several subplots with different partition.


Starting from the code below, try to reproduce the graphic on the right.

::

   plt.subplot(2, 2, 1)
   plt.subplot(2, 2, 3)
   plt.subplot(2, 2, 4)

Click on figure for solution.


Polar Axis
----------

.. image:: auto_examples/images/plot_polar_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_polar_ex.html


.. hint::

   You only need to modify the ``axes`` line


Starting from the code below, try to reproduce the graphic on the right.

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

Click on figure for solution.


3D Plots
--------

.. image:: auto_examples/images/plot_plot3d_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_plot3d_ex.html


.. hint::

   You need to use `contourf
   <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.contourf>`_


Starting from the code below, try to reproduce the graphic on the right.

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

Click on figure for solution.

.. seealso:: :ref:`mayavi-label`

Text
----


.. image:: auto_examples/images/plot_text_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_text_ex.html


.. hint::

   Have a look at the `matplotlib logo
   <http://matplotlib.org/examples/api/logo2.html>`_.

Try to do the same from scratch !

Click on figure for solution.

____


.. topic:: **Quick read**

   If you want to do a first quick pass through the Scipy lectures to
   learn the ecosystem, you can directly skip to the next chapter:
   :ref:`scipy`.

   The remainder of this chapter is not necessary to follow the rest of
   the intro part. But be sure to come back and finish this chapter later.

Beyond this tutorial
====================

Matplotlib benefits from extensive documentation as well as a large
community of users and developers. Here are some links of interest:

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

* `User guide <http://matplotlib.org/users/index.html>`_

* `FAQ <http://matplotlib.org/faq/index.html>`_

  - Installation
  - Usage
  - How-To
  - Troubleshooting
  - Environment Variables

* `Screenshots <http://matplotlib.org/users/screenshots.html>`_


Code documentation
------------------

The code is well documented and you can quickly access a specific command
from within a python session:

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

The `matplotlib gallery <http://matplotlib.org/gallery.html>`_ is
also incredibly useful when you search how to render a given graphic. Each
example comes with its source.


Mailing lists
--------------

Finally, there is a `user mailing list
<https://mail.python.org/mailman/listinfo/matplotlib-users>`_ where you can
ask for help and a `developers mailing list
<https://mail.python.org/mailman/listinfo/matplotlib-devel>`_ that is more
technical.


Quick references
================

Here is a set of tables that show main properties and styles.

Line properties
----------------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Property
     - Description
     - Appearance

   * - alpha (or a)
     - alpha transparency on 0-1 scale
     - .. image:: auto_examples/images/plot_alpha_1.png

   * - antialiased
     - True or False - use antialised rendering
     - .. image:: auto_examples/images/plot_aliased_1.png
       .. image:: auto_examples/images/plot_antialiased_1.png

   * - color (or c)
     - matplotlib color arg
     - .. image:: auto_examples/images/plot_color_1.png

   * - linestyle (or ls)
     - see `Line properties`_
     -

   * - linewidth (or lw)
     - float, the line width in points
     - .. image:: auto_examples/images/plot_linewidth_1.png

   * - solid_capstyle
     - Cap style for solid lines
     - .. image:: auto_examples/images/plot_solid_capstyle_1.png

   * - solid_joinstyle
     - Join style for solid lines
     - .. image:: auto_examples/images/plot_solid_joinstyle_1.png

   * - dash_capstyle
     - Cap style for dashes
     - .. image:: auto_examples/images/plot_dash_capstyle_1.png

   * - dash_joinstyle
     - Join style for dashes
     - .. image:: auto_examples/images/plot_dash_joinstyle_1.png

   * - marker
     - see `Markers`_
     -

   * - markeredgewidth (mew)
     - line width around the marker symbol
     - .. image:: auto_examples/images/plot_mew_1.png

   * - markeredgecolor (mec)
     - edge color if a marker is used
     - .. image:: auto_examples/images/plot_mec_1.png

   * - markerfacecolor (mfc)
     - face color if a marker is used
     - .. image:: auto_examples/images/plot_mfc_1.png

   * - markersize (ms)
     - size of the marker in points
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

All colormaps can be reversed by appending ``_r``. For instance, ``gray_r`` is
the reverse of ``gray``.

If you want to know more about colormaps, checks `Documenting the matplotlib
colormaps <intro/matplotlib/matplotlib.rst>`_.

.. image:: auto_examples/images/plot_colormaps_1.png
   :scale: 80

