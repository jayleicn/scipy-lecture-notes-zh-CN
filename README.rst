.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.31521.svg
    :target: http://dx.doi.org/10.5281/zenodo.31521

.. image:: https://travis-ci.org/scipy-lectures/scipy-lecture-notes.svg?branch=master
    :target: https://travis-ci.org/scipy-lectures/scipy-lecture-notes

===================
Scipy-Lecture-Notes
===================

这是来自 http://scipy-lectures.org 的Python科学计算环境教程的中文版

这些文件使用rest markup语言写作 (后缀名为 ``.rst`` ) 并使用Sphinx编译: http://sphinx.pocoo.org/.

在线地址: http://scipy-lectures.cn

临时地址: http://115.28.54.204


使用 && 分发
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


致谢
--------------------------

感谢

原作者及编辑者们，
最初发起这个计划的@teastares童鞋，
所有为译文作出贡献的童鞋。

