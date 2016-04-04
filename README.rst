.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.31521.svg
    :target: http://dx.doi.org/10.5281/zenodo.31521

.. image:: https://travis-ci.org/scipy-lectures/scipy-lecture-notes.svg?branch=master
    :target: https://travis-ci.org/scipy-lectures/scipy-lecture-notes

===================
Scipy-Lecture-Notes
===================

这是来自 http://scipy-lectures.org 的Python科学计算环境教程的中文版

这些文件使用rest markup语言写作 (后缀名为 ``.rst`` ) 并使用Sphinx编译: http://sphinx.pocoo.org/.

网页地址: http://scipy-lectures.cn


授权许可
-------------------------

如 ``LICENSE.txt`` 文件指出,这份材料无任何附加条款。基于教学目的，您可以自由使用和更改。

但是，为了使这份材料能够不断的改进，我们鼓励大家将自己所做的更改反馈到此处，以期原作者，编辑以及译者能够持续优化此文档。


编译 
--------------------------

在ubuntu14.04 LTS上编译HTML (我的编译环境)

1. 安装anaconda for Python 2.7, Linux 64-bit <https://www.continuum.io/downloads>. (其他版本未经测试)
2. 添加anaconda/bin/目录到系统PATH变量中  
3. 安装seaborn包  pip install seaborn
4. 切换至scipy-lecture-notes-zh-CN/目录下， make html
5. 在build/html文件夹下打开index.html


如何做出贡献
---------------------------------------

如果你对翻译此文档感兴趣，请fork此仓库，翻译或修改之后pull request。 在线编辑，搭配实时预览，体验更佳!

文档 ``CONTRIBUTING.rst`` 包含更详细的指导。(包含编译和贡献部分)


翻译进程及译者概况
---------------------------------------

1. 基础教程

1.1. 科学计算工具和流程  @jayleicn

1.2. Python 语言  @teastares

1.3. NumPy: 操作和处理数据  @teastares

1.4. Matplotlib: 绘图 @jayleicn

1.5. Scipy : high-level scientific computing

1.6. Getting help and finding documentation

2. Advanced topics

2.1. Advanced Python Constructs

2.2. Advanced Numpy

2.3. Debugging code

2.4. Optimizing code

2.5. Sparse Matrices in SciPy

2.6. Image manipulation and processing using Numpy and Scipy

2.7. Mathematical optimization: finding minima of functions

2.8. Interfacing with C

3. Packages and applications

3.1. Statistics in Python

3.2. Sympy : Symbolic Mathematics in Python

3.3. Scikit-image: image processing

3.4. Traits: building interactive dialogs

3.5. 3D plotting with Mayavi

3.6. scikit-learn: machine learning in Python




致谢
--------------------------

感谢

原作者及编辑者们，
最初发起这个计划的@teastares童鞋，
所有为译文作出贡献的童鞋。

