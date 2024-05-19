#include <iostream>
#include <pthread.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h> 
const int n = 400;
using namespace std;
float A[n][n];
int worker_count = 2; //工作线程数量
void init()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = 0;
        }
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            A[i][j] = rand() % 100;
    }

    for (int i = 0; i < n; i++)
    {
        int k1 = rand() % n;
        int k2 = rand() % n;
        for (int j = 0; j < n; j++)
        {
            A[i][j] += A[0][j];
            A[k1][j] += A[k2][j];
        }
    }
}

void f_ordinary()
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

struct threadParam_t
{
    int k; //消去的轮次
    int t_id; // 线程 id
};

void* threadFunc(void* param)
{

    __m256 va, vt, vx, vaij, vaik, vakj;

    threadParam_t* p = (threadParam_t*)param;
    int k = p->k; //消去的轮次
    int t_id = p->t_id; //线程编号
    int i = k + t_id + 1; //获取自己的计算任务

    for (int m = k + 1 + t_id; m < n; m += worker_count)
    {
        // 初始化一个AVX向量来存储A[m][j]的当前值  
        __m256 Amj_vec = _mm256_setzero_ps(); // 假设先置零，实际中可能需要加载数据  

        // 处理内层循环的向量部分（每次8个元素）  
        for (int j = k + 1; j < n - 7; j += 8) // 确保不会越界，并且每次增加8  
        {
            // 加载A[k][j]到AVX向量  
            __m256 Akj_vec = _mm256_loadu_ps(&A[k][j]);

            // 如果Amj_vec还没有被加载（例如，这是m的第一个迭代），则加载A[m][j]  
            if (m == k + 1 + t_id) {
                Amj_vec = _mm256_loadu_ps(&A[m][j]);
            }

            __m256 result_vec = _mm256_sub_ps(Akj_vec, _mm256_mul_ps(Akj_vec, Amj_vec));

            // 存储结果到A[i][j]  
            _mm256_storeu_ps(&A[i][j], result_vec);
        }

        // 处理剩余的少于8个元素的列（如果有的话）  
        for (int j = n - 7; j < n; j++)
        {
            A[i][j] = A[k][j] - A[k][j] * A[m][j]; // 使用标量操作  
        }

        // 设置A[m][k]为0（不依赖于j的循环）  
        A[m][k] = 0.0f;
    }
    pthread_exit(NULL);
}

int main()
{
    init();

    clock_t  clockBegin, clockEnd;
    clockBegin = clock(); //开始计时

    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }

        //创建工作线程，进行消去操作

        pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
        threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构

        //分配任务
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //创建线程
        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);

        //主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_join(handles[t_id], NULL);

    }

    clockEnd = clock();
    cout << "总共耗时： " << clockEnd - clockBegin << "ms" << endl;
    system("pause");
    return 0;
}
