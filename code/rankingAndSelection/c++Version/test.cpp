#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<numeric>
#include<random>

using namespace std;
random_device rd;
uniform_real_distribution<float> distribution(0,1);
mt19937_64 generator(rd());
// extern "C" int hehe(int i, int j);
// int add(int a){
//     return ++a;
// }
// int hehe(int i, int j){
//         // int j = 0;
//         // for (int i = 0; i < 1000000000; i++)
//         // {
//         //     j ++;
//         // }
//         cout<<add(i);
//         cout<<add(i) + add(j)<<endl;
//         return 0;
// }
//extern "C" class Test;
class Test 
{  
    private:
    int a;
public: 
 void set(){
     a = 1;
 }
 int get(){
     return a+3;
 }
 int Add(const int x, const int y) 
 { 
  return x + y; 
 } 
  
 int Del(const int x, const int y) 
 { 
  return x - y; 
 } 
}; 
// extern "C" int run(int i, int j);
// int run(int i, int j){
//     Test t;
//     cout<<t.Del(i,j)<<endl;
//     return 0;
// }

// extern "C" int run(float *i);
// int run(float *i){
//     cout<<i[1]<<endl;
//     return 0;
// }
extern "C" int run();
int run(){
    Test t;
    t.set();
    cout<<t.get();
    return 0;
}


#include <iostream>
#include <queue>
using namespace std;

//方法1
struct tmp1 //运算符重载<
{
    float x;
    int y;
    bool operator() (tmp1 a, tmp1 b) 
    {
        return a.x > b.x; //小顶堆
    }
};

int main() 
{
    int a[3];
    for (int i=0;i<3;i++){
        if (i<1){
            a[i] = 0;
            cout<<a[i]<<endl;
        }else{
            a[i]=1;
        }
        
    }
    cout<<a[2]<<endl;
    // tmp1 a = {1.2, 1};
    // tmp1 b = {2.2,1};
    // tmp1 c = {3.2,2};
    // priority_queue<tmp1, vector<tmp1>, tmp1 > d;
    // d.push(b);
    // d.push(c);
    // d.push(a);
    // while (!d.empty()) 
    // {
    //     //cout << d.top().x << '\n';
    //     d.pop();
    // }
    // cout << endl;

    // float e[] = {1,2,3};
    // for (float i : e){
    //     i++;
    // }
    // vector<exponential_distribution<float> > f;
    // for (int i=1;i<3;i++){
    //     exponential_distribution<float> a(i);
    //     f.push_back(a);
    // }
    // //cout<<f[0](generator)<<endl;
    // int l[5][5]={0};
    // for(int i=0;i<=5;i++)
    // {
    // for(int j=0;j<=5;j++)
    // {
    // //cout<<l[i][j];
    // }
    // cout<<endl;
    // }
}