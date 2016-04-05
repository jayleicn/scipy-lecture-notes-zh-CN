===========================
NumPy: 创建与操作数值数据
===========================

**作者**：*Emmanuelle Gouillart, Didrik Pinte, Gael Varoquaux, and Pauli Virtanen*

**译者**：*ChzRuan*

本章对Python数值计算的高性能核心工具 --- NumPy模块进行综述。

-------------------------
NumPy的数组(array)对象
-------------------------

.. contents:: 本节内容
    :local:
    :depth: 1

NumPy及NumPy数组(array) 是啥？
-------------------------------

NumPy 数组(array)
..................

    - **Python自带：**
        - 高级数值对象：整数，浮点数
        - 容器：列表(list, 快速插入/追加元素)，字典(dictionary, 快速检索)
    - **NumPy提供：**
        - 多维数组
        - 更接近硬件端（高效）
        - 为科学计算设计（便利）
        - 面向数组的计算思想

.. sourcecode:: ipython

    >>> import numpy as np
    >>> a = np.array([0, 1, 2, 3])
    >>> a
    array([0, 1, 2, 3])

一个NumPy数组的内容可以是：
    - 实验/仿真数据（关于时间的分立序列）
    - 测量装置的采样信号。例：声波
    - 图片（灰度图或色图）的像素点
    - 在不同的X-Y-Z位置观测的3-D数据。例：核磁共振扫描
    - ...

**NumPy数组的优点**：它的内存管理更高效，因此数值计算速度更快。

.. sourcecode:: ipython

    In [1]: L = range(1000)
    
    In [2]: %timeit [i**2 for i in L]
    1000 loops, best of 3 : 403 us per loops
    
    In [3]: a = np.arange(1000)
    
    In [4]: %timeit a**2
    100000 loops, best of 3 : 12.7 us per loops

NumPy帮助文档
..............

    - 网址： http://docs.scipy.org/
    - 交互式帮助：
    .. sourcecode:: ipython

       In [5]: np.array?
        String Form:<built-in function array>
        Docstring:
        array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0, ...


    - 寻找某物：
    
    .. sourcecode:: ipython

        >>> np.lookfor('create array')
        Search results for 'create array'
        ----------------------------------
        numpy.array
            Create an array
        numpy.memmap
            Create a memory-map to an array stored in a *binary* file on a disk

.. sourcecode:: ipython
    In [6]: np.con*?
    np.concatenate
    np.conj
    np.conjugate
    np.convolve
