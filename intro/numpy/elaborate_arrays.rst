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


Structured data types
---------------------

=============== ====================
``sensor_code``  (4-character string)
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

Field access works by indexing with field names::

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

Multiple fields at once::

    >>> samples[['position', 'value']]
    array([(1.0, 0.37), (1.0, 0.11), (1.0, 0.13), (1.5, 0.37), (3.0, 0.11),
           (1.2, 0.13)], 
          dtype=[('position', '<f8'), ('value', '<f8')])

Fancy indexing works, as usual::

    >>> samples[samples['sensor_code'] == 'ALFA']    # doctest: +SKIP
    array([('ALFA', 1.5, 0.37), ('ALFA', 3.0, 0.11)], 
          dtype=[('sensor_code', 'S4'), ('position', '<f8'), ('value', '<f8')])

.. note:: There are a bunch of other syntaxes for constructing structured
   arrays, see `here <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`__
   and `here <http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#specifying-and-constructing-data-types>`__.


:class:`maskedarray`: dealing with (propagation of) missing data
------------------------------------------------------------------

* For floats one could use NaN's, but masks work for all types::

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

* Masking versions of common functions::

    >>> np.ma.sqrt([1, -1, 2, -2]) #doctest:+ELLIPSIS
    masked_array(data = [1.0 -- 1.41421356237... --],
                 mask = [False  True False  True],
           fill_value = 1e+20)
    <BLANKLINE>


.. note::

   There are other useful :ref:`array siblings <array_siblings>`


_____

While it is off topic in a chapter on numpy, let's take a moment to
recall good coding practice, which really do pay off in the long run:

.. topic:: Good practices

    * Explicit variable names (no need of a comment to explain what is in
      the variable)

    * Style: spaces after commas, around ``=``, etc.

      A certain number of rules for writing "beautiful" code (and, more
      importantly, using the same conventions as everybody else!) are
      given in the `Style Guide for Python Code
      <https://www.python.org/dev/peps/pep-0008>`_ and the `Docstring
      Conventions <https://www.python.org/dev/peps/pep-0257>`_ page (to
      manage help strings).

    * Except some rare cases, variable names and comments in English.


