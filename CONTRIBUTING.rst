帮助我们
=============

编译指南
----------------------

编译HTML格式文件, 输入::

    make html

生成的HTML文件保存在 ``build/html`` 目录下

首次编译耗时较长大致3~10分钟，这取决于编译平台的硬件配置及网络环境。由于第一次编译的信息缓存，随后的编译速度会稍快。


生成PDF文档::

    make pdf

生成过程中可能产生一些TeX相关的错误. 细微调整 ``*.rst`` 文档的排版通常有助于改善这些错误.


需求
............

*可能不完全*

* make
* sphinx (>= 1.0)
* pdflatex
* pdfjam
* matplotlib
* scikit-learn (>= 0.8)
* scikit-image
* pandas
* seaborn


在ubuntu14.04 LTS上编译HTML (我的编译环境)
------------------

1. 安装anaconda for Python 2.7, Linux 64-bit <https://www.continuum.io/downloads>. (其他版本未经测试)
2. 添加anaconda/bin/目录到系统PATH变量中  
3. 安装seaborn包  pip install seaborn
4. 切换至scipy-lecture-notes-zh-CN/目录下， make html


在Fedora上编译 (原作者)
------------------

As root::

    yum install python make python-matplotlib texlive-pdfjam texlive scipy \ 
    texlive-framed texlive-threeparttable texlive-wrapfig texlive-multirow
    pip install Sphinx
    pip install Cython
    pip install scikit-learn
    pip install scikit-image


如何作出贡献
---------------------------------------

如果你对翻译此文档感兴趣，请fork此仓库，在作出翻译之后向我pull request。 感谢!

