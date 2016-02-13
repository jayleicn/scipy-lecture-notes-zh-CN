Scipy Lecture Notes 
===========================

.. only:: html

   一份文档，帮助你熟悉Python的数值，科学以及数据处理生态。
   --------------------------------------------------------------

.. raw html to center the title

.. raw:: html

  <style type="text/css">
    div.documentwrapper h1 {
        text-align: center;
        font-size: 280% ;
        font-weight: bold;
        margin-bottom: 4px;
    }

    div.documentwrapper h2 {
        background-color: white;
        border: none;
        font-size: 130%;
        text-align: center;
        margin-bottom: 40px;
        margin-top: 4px;
    }

    a.headerlink:after {
        content: "";
    }

    div.sidebar {
        margin-right: -20px;
        margin-top: -10px;
        border-radius: 6px;
        font-family: FontAwesome, sans-serif;
        min-width: 200pt;
    }

    div.sidebar ul {
        list-style: none;
        text-indent: -3ex;
        color: #555;
    }

    div.sidebar li {
        margin-top: .5ex;
    }

    div.preface {
        margin-top: 20px;
    }

  </style>

.. nice layout in the toc

.. include:: tune_toc.rst 

.. |pdf| unicode:: U+f1c1 .. PDF file

.. |archive| unicode:: U+f187 .. archive file

.. |github| unicode:: U+f09b  .. github logo

.. only:: html

    .. sidebar::  下载 
       
       * |pdf| `PDF, 双页 <./_downloads/ScipyLectures.pdf>`_

       * |pdf| `PDF, 单页 <./_downloads/ScipyLectures-simple.pdf>`_
   
       * |archive| `HTML格式 <https://github.com/jayleicn/www.scipy-lectures.cn/archive/master.zip>`_
     
       * |github| `源码 (github) <https://github.com/jayleicn/scipy-lecture-notes-zh-CN/edit/master/index.rst>`_


    这是一份Python数据处理的指南。它包括，但不仅限于Python科学计算，数据处理相关的核心工具和技巧。
    它由易到难，每一个章节都对应着1~2个小时的课程。我们相信不同水平的读者都可以从这份指南中获益。

    .. rst-class:: preface

        .. toctree::
            :maxdepth: 2

            preface.rst

|

.. rst-class:: tune

  .. toctree::
    :numbered:

    intro/index.rst
    advanced/index.rst
    packages/index.rst

|

..  
 FIXME: I need the link below to make sure the banner gets copied to the
 target directory.

.. only:: html

 .. raw:: html
 
   <div style='display: none; height=0px;'>

 :download:`ScipyLectures.pdf` :download:`ScipyLectures-simple.pdf`
 
 .. image:: themes/plusBox.png

 .. image:: images/logo.svg

 .. raw:: html
 
   </div>
   </small>


..
    >>> # For doctest on headless environments (needs to happen early)
    >>> import matplotlib
    >>> matplotlib.use('Agg')




