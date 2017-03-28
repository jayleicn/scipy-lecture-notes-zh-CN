===================
Scipy-Lecture-Notes-CN
===================

这是来自 [scipy-lectures.org](http://scipy-lectures.org) 的Python科学计算环境教程的中文版

在线版不再提供, 请在[releases](https://github.com/jayleicn/scipy-lecture-notes-zh-CN/releases)下载网页格式的离线版文档, 打开index.html浏览教程

授权许可
-------------------------

参见 ``LICENSE.txt`` 


编译 
--------------------------

在ubuntu14.04 LTS上编译HTML (我的编译环境)

1. 安装[anaconda](https://www.continuum.io/downloads) for Python 2.7, Linux 64-bit. (其他版本未经测试), 并将anaconda python添加到系统PATH变量中

2. 安装seaborn包

   ```
   pip install seaborn
   ```

3. 切换至scipy-lecture-notes-zh-CN/目录下， make html

4. 在build/html文件夹下打开index.html


注意: anaconda中默认安装的docutils版本会导致编译错误, 请在编译之前运行此命令

```
pip install docutils==0.12
```

如何做出贡献
---------------------------------------

如果你对翻译此文档感兴趣, 请fork此仓库, 翻译或修改之后pull request. 


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

原作者及编辑者们,
最初发起这个计划的@teastares童鞋,
所有为译文作出贡献的童鞋.

