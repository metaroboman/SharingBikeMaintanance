#include<iostream>
#include<string>
#include<random>
#include<queue>
#include<vector>
//#include<numeric>

using namespace std;

random_device rd;
mt19937_64 generator(rd());


typedef struct event //运算符重载<
    {
        float t;
        int k; // kind
        int stt; // start 
        int ter; // terminal
        bool operator() (event a, event b) 
        {
            return a.t > b.t; //小顶堆
        }
    };

class Solution {
public:
    vector<int> psum;
    int tot = 0;
    //c++11 random integer generation
    uniform_int_distribution<int>  uni;

    Solution(vector<float> w) {
        for (float x : w) {
            tot += x;
            psum.push_back(tot);
        }
        uniform_real_distribution<int>  uni(0, 1);
    }

    int pickIndex() {
        int targ = uni(generator);

        int lo = 0, hi = psum.size() - 1;
        while (lo != hi) {
            int mid = (lo + hi) / 2;
            if (targ >= psum[mid]) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
};


static const int A = 2;
    static const int M = 6;
    static float Pij[A*A] = {0.1,0.1,0.1,0.1};
    static float ArrLst[A] = {0.5, 0.5};
    static float RhoMtx[A*A] = {0.1,0.1,0.1,0.1};
    static float Beta = 0.3;
    static float Tau = 1.0;
    static int C = 1;
    static float Mu = 1.0;
    static int N = 1;
    static int TimeLimit = 10000;
    static int NumDis[A] = {2,4};
    static int ArrRank[A] = {1,0};
float T =0.0;

priority_queue<event, vector<event>, event > F;
priority_queue<event, vector<event>, event > S;
vector<exponential_distribution<float> > f;
vector<exponential_distribution<float> > c_exp;
vector< exponential_distribution<float> > r;
vector<exponential_distribution<float> > m;
vector<uniform_real_distribution<float> > uni;

vector<Solution> selector;


int state1[A+3] = {0};
int **state2;
int arrival = 0;
int lost = 0;

int kind; int start; int terminal;


void reset(){
  
    
    for (int i=0;i<A;i++){
        exponential_distribution<float> a(ArrLst[i]);
        f.push_back(a);
    }

    exponential_distribution<float> c_e(1);
    c_exp.push_back(c_e);
    
    for (int i=0;i<A;i++){
        for (int j=0;j<A;j++){
            exponential_distribution<float> b(RhoMtx[i*A+j]);
            r.push_back(b);
        }
    }

    exponential_distribution<float> c(Mu);
    m.push_back(c);

    uniform_real_distribution<float> d(0,1);
    uni.push_back(d);

    for (int i=0;i<A;i++){
        vector<float> weight;
        for (int j=0;j<A;j++){
            weight.push_back(Pij[i*A+j]);
        }
        Solution s(weight);
        selector.push_back(s);
    }
    
    // scheduler
    for (int i=0; i<A; i++){
        event E = {f[i](generator), -1, i, i};
        S.push(E);
    }
    // scheduler for carrier
    for (int i=0; i<C; i++){
        event c = {c_exp[0](generator), 2, i, i};
        S.push(c); 
    }
    // state1
    int average = int(M/A);
    for (int i=0; i<A+3; i++){
        if (i<A) {
            //cout<<i<<endl;
            state1[i] = average;
        }else{
            state1[i] = 0;
        }
    }
    // state2
    for (int i=0;i<A;i++) { for (int j=0;j<A;j++){state2[i][j] = 0;}}

    cout<<f[0](generator)<<endl;
}

void set_record(int kind){
    if (T > 0.8 * TimeLimit){
        if (kind == -10) {
            arrival++;
        }else if (kind == -11)
        {
            lost++;
        }
    }
}


void add_event(int kind, int start, int terminal){
    float next_time = 0.0;
    int s,e;
    if (kind == -1){
        next_time = f[start](generator) + T;
        s= start;e = start;
    }
    else if (kind == 1){
        next_time = r[start*A+terminal](generator) + T;
        s = start; e = terminal;
    }
    else if (kind == 2){ 
        next_time = c_exp[0](generator) + T;
        s = -3; e = -2;
    }
    else if( kind == 3){
        next_time = m[0](generator);
        if (state1[A+1] < N){
            next_time += T;
            event E = {next_time, 0, 0, 0};
            F.push(E);
        }
        else{
            next_time += F.top().t;
            F.pop();
            event E = {next_time, 0, 0, 0};
            F.push(E);
        }
        s = -2; e = -1;
    }
    else{
        next_time = c_exp[0](generator);
        next_time += T;
        s = -1; e = 0;
    }
    event E = {next_time, kind, s, e};
    S.push(E);
}

void bikeArr(){
    state2[start][terminal] -= 1;
    S.pop();
    if (uni[0](generator)<Beta){
        state1[A] += 1;
    }
    else{
        state1[terminal] += 1;
    }
}
void BPover(){
    S.pop();
    int k = min(state1[A], C);
    for (int i=0;i<k;i++){
        add_event(3, start, terminal);
        state1[A]--;
        state1[A+1]++;
    }
    add_event(2, start, terminal);
}
void repair(){
    S.pop();
    if (state1[A+1] <= N){
        F.pop();
    }
    state1[A+1]--;
    state1[A+2]++;
}
void DPover(){
    S.pop();
    int k = min(state1[A+2], C);
    state1[A+2]-=k;
    int alloc = 0;
    for (int i=0;i<A;i++){
        int n = ArrRank[i];
        if (state1[n]>NumDis[n]) continue;
        else{
            alloc = min(NumDis[n]-state1[i], k);
            state1[n] += alloc;
            k -= alloc;
            if (k==0) break;
        }
    }
    add_event(4,start,terminal);
}
void curArr(){
    set_record(-10);
    if (state1[start] == 0){
        S.pop();
        add_event(-1, start, terminal);
    }else{
        S.pop();
        add_event(1, start, terminal);
        terminal = 1;//selector[start].pickIndex();
        state1[start] -= 1;
        state2[start][terminal] += 1;
        add_event(1, start, terminal);
    }
}
void stepForward(){
    event E = S.top();
    S.pop();
    T = E.t; kind = E.k; start = E.stt; terminal = E.ter;
    if (kind == 1){
        bikeArr();
    }else if (kind == 2){
        BPover();
    }else if (kind == 3){
        repair();
    }else if (kind == 4){
        DPover();
    }else {
        curArr();
    }
    
}
float simulate(){
    reset();
    while (T < TimeLimit){
        stepForward();
    }
    return lost / arrival;
}

int main(){
    
    //float pij = 0.2;
    //cout<<T<<endl;
    cout<<simulate()<<endl;
    //t.Print();
    //exponential_distribution<float> ed(1);
    //cout<<(ed(1));
    return 0;
}