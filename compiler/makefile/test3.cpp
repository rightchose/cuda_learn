#include <iostream>
using namespace std;

extern "C" int func(); // 注意这里的声明
int main()
{
    func();
    return 0;
}