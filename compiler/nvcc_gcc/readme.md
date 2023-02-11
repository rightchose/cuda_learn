#### 方法一
将所有文件分别编译，最后统一合并。
对于C程序
```
nvcc -c test1.cu
gcc -c test2.c # C程序调用
gcc -o testc test1.o test2.o -lcudart -L/usr/local/cuda/lib64 -lstdc++
```
对于C++程序
```
nvcc -c test1.cu
g++ -c test3.cpp
g++ -o testcpp test1.0 test3.o -lcudart -L/usr/local/cuda/lib64
```

#### 方案二
将CUDA程序变成静态库。
对于C程序
```
nvcc -lib test1.cu -o libtestcu.a
gcc test2.c -ltestcu -L. -lcudart -L/usr/local/cuda/lib64 -o testc -lstdc++
```
对于C++
```
nvcc -lib test1.cu -o libtestcu.a
g++ test3.cpp -ltestcu -L. -lcudart -L/usr/local/cuda/lib64 -o testcpp
```

#### 方案三
将CUDA程序弄成动态库
使用makefile


