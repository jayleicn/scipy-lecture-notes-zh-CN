.. For doctests

   >>> import numpy as np
   >>> np.random.seed(0)
   >>> from matplotlib import pyplot as plt

.. currentmodule:: numpy

数组内容进阶
======================

.. contents:: Section contents
    :local:
    :depth: 1

更多数据类型
---------------

Casting
........

不同类型的数组合体运算，所得结果的类型是“范围”更大的那个::

    >>> np.array([1, 2, 3]) + 1.5
    array([ 2.5,  3.5,  4.5])

赋值永远不会改变数据类型！::

    >>> a = np.array([1, 2, 3])
    >>> a.dtype
    dtype('int64')
    >>> a[0] = 1.9     # <-- 浮点型被截断为整型
    >>> a
    array([1, 2, 3])

强制cast::

    >>> a = np.array([1.7, 1.2, 1.6])
    >>> b = a.astype(int)  # <-- 截断为整型
    >>> b
    array([1, 1, 1])

舍入为整数::

    >>> a = np.array([1.2, 1.5, 1.6, 2.5, 3.5, 4.5])
    >>> b = np.around(a)
    >>> b                    # 仍为浮点数
    array([ 1.,  2.,  2.,  2.,  4.,  4.])
    >>> c = np.around(a).astype(int)
    >>> c
    array([1, 2, 2, 2, 4, 4])

不同数据类型的大小
..........................

Integers (signed):
整型（有符号）:

=================== ==============================================================
:class:`int8`        8 bits
:class:`int16`       16 bits
:class:`int32`       32 bits (与32位系统的 :class:`int` 相同)
:class:`int64`       64 bits (与64位系统的 :class:`int` 相同)
=================== ==============================================================

::

    >>> np.array([1], dtype=int).dtype
    dtype('int64')
    >>> np.iinfo(np.int32).max, 2**31 - 1
    (2147483647, 2147483647)


无符号整型:

=================== ==============================================================
:class:`uint8`       8 bits
:class:`uint16`      16 bits
:class:`uint32`      32 bits
:class:`uint64`      64 bits
=================== ==============================================================

::

    >>> np.iinfo(np.uint32).max, 2**32 - 1
    (4294967295, 4294967295)

.. sidebar:: 长整型

    Python 2 有特定的长整型类型，它不会溢出，在数字后面加'L'表示。在Python 3，所有的整型都是长整型，因此都不会溢出。

     >>> np.iinfo(np.int64).max, 2**63 - 1  # doctest: +SKIP
     (9223372036854775807, 9223372036854775807L)


浮点数 :

=================== ==============================================================
:class:`float16`     16 bits
:class:`float32`     32 bits
:class:`float64`     64 bits (与 :class:`float` 相同)
:class:`float96`     96 bits, 依赖于系统 (与 :class:`np.longdouble` 相同)
:class:`float128`    128 bits, 依赖于系统 (与 :class:`np.longdouble` 相同)
=================== ==============================================================

::

    >>> np.finfo(np.float32).eps
    1.1920929e-07
    >>> np.finfo(np.float64).eps
    2.2204460492503131e-16

    >>> np.float32(1e-8) + np.float32(1) == 1
    True
    >>> np.float64(1e-8) + np.float64(1) == 1
    False

浮点复数：

=================== ==============================================================
:class:`complex64`   两个 32-bit 浮点数
:class:`complex128`  两个 64-bit 浮点数
:class:`complex192`  两个 96-bit 浮点数，依赖于系统
:class:`complex256`  两个 128-bit 浮点数，依赖于系统
=================== ==============================================================

.. topic:: 不常用的数据类型

   除非已确认必须要使用特定的数据类型，大多数情况使用默认类型即可。

   例如，若使用 ``float32`` 类型而非默认的 ``float64`` ，则：

   - 磁盘内存用量减半
   - 所需内存带宽减半（使得某些运算更快）

     .. sourcecode:: ipython

        In [1]: a = np.zeros((1e6,), dtype=np.float64)

        In [2]: b = np.zeros((1e6,), dtype=np.float32)

        In [3]: %timeit a*a
        1000 loops, best of 3: 1.78 ms per loop

        In [4]: %timeit b*b
        1000 loops, best of 3: 1.07 ms per loop

   - 但是：这会产生更大的舍入误差 --- 有时这会造成严重后果（因而，除非确实需要 ``float32`` ，尽量采用默认类型）。


结构化数据类型
---------------------

结构化数据类型例：
=============== ====================
``sensor_code``(4-character string)
``position``     (float)
``value``        (float)
=============== ====================

::

    >>> samples = np.zeros((6,), dtype=[('sensor_code', 'S4'),
    ...                                 ('position', float), ('value', float)])
    >>> samples.ndim
    1
    >>> samples.shape
    (6,)
    >>> samples.dtype.names
    ('sensor_code', 'position', 'value')

    >>> samples[:] = [('ALFA',   1, 0.37), ('BETA', 1, 0.11), ('TAU', 1,   0.13),
    ...               ('ALFA', 1.5, 0.37), ('ALFA', 3, 0.11), ('TAU', 1.2, 0.13)]
    >>> samples     # doctest: +SKIP
    array([('ALFA', 1.0, 0.37), ('BETA', 1.0, 0.11), ('TAU', 1.0, 0.13),
           ('ALFA', 1.5, 0.37), ('ALFA', 3.0, 0.11), ('TAU', 1.2, 0.13)], 
          dtype=[('sensor_code', 'S4'), ('position', '<f8'), ('value', '<f8')])

可以通过域名索引实现域名访问::

    >>> samples['sensor_code']    # doctest: +SKIP
    array(['ALFA', 'BETA', 'TAU', 'ALFA', 'ALFA', 'TAU'], 
          dtype='|S4')
    >>> samples['value']
    array([ 0.37,  0.11,  0.13,  0.37,  0.11,  0.13])
    >>> samples[0]    # doctest: +SKIP
    ('ALFA', 1.0, 0.37)

    >>> samples[0]['sensor_code'] = 'TAU'
    >>> samples[0]    # doctest: +SKIP
    ('TAU', 1.0, 0.37)

也可以一次访问多个域名::

    >>> samples[['position', 'value']]
    array([(1.0, 0.37), (1.0, 0.11), (1.0, 0.13), (1.5, 0.37), (3.0, 0.11),
           (1.2, 0.13)], 
          dtype=[('position', '<f8'), ('value', '<f8')])

之前各种索引技巧也可用于域名索引::

    >>> samples[samples['sensor_code'] == 'ALFA']    # doctest: +SKIP
    array([('ALFA', 1.5, 0.37), ('ALFA', 3.0, 0.11)], 
          dtype=[('sensor_code', 'S4'), ('position', '<f8'), ('value', '<f8')])

.. note:: 构造结构化数组的其他语法可见 `这儿 <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`__
   与 `这儿 <http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#specifying-and-constructing-data-types>`__.


:class:`maskedarray`: 处理（或传递）缺失数据
------------------------------------------------------------------

* 对浮点数数组略去某些元素可以利用NaN（Not-a-Number）元素，而掩码（mask）技巧对所有类型的数组都适用::

    >>> x = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 1])
    >>> x
    masked_array(data = [1 -- 3 --],
                 mask = [False  True False  True],
           fill_value = 999999)
    <BLANKLINE>

    >>> y = np.ma.array([1, 2, 3, 4], mask=[0, 1, 1, 1])
    >>> x + y
    masked_array(data = [2 -- -- --],
                 mask = [False  True  True  True],
           fill_value = 999999)
    <BLANKLINE>

* 掩码对一般的函数也有效::

    >>> np.ma.sqrt([1, -1, 2, -2]) #doctest:+ELLIPSIS
    masked_array(data = [1.0 -- 1.41421356237... --],
                 mask = [False  True False  True],
           fill_value = 1e+20)
    <BLANKLINE>


.. note::

   除数组外，其他有用的数据结构可见 :ref:`array siblings <array_siblings>`。


_____

下面的内容在介绍NumPy的章节中显的有点离题，我们讨论一些好的代码习惯，养成这一优良习惯会使未来受益无穷。


.. topic:: 优良代码习惯

    * 有意义的变量名（不需要注释来解释变量的内容是啥）

    * 风格：在逗号后面、等号两侧等合适的地方加空格。

      更多“优美”代码的规范（比优美更重要的是其他人也遵循这些惯例！）可见 `Style Guide for Python Code
      <https://www.python.org/dev/peps/pep-0008>`_ 以及 `Docstring
      Conventions <https://www.python.org/dev/peps/pep-0257>`_ （它介绍字符串处理）。

    * 变量名、注释用英文，除非极端情况。


