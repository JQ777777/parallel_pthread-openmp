#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h> 
using namespace std;

const int n = 400;
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
    int t_id; // 线程 id
};

//信号量定义
sem_t sem_main;
sem_t* sem_workerstart = new sem_t[worker_count];//每个线程有自己专属的信号量
sem_t* sem_workerend = new sem_t[worker_count];

void* threadFunc(void* param)
{
    threadParam_t* p = (threadParam_t*)param;

    int t_id = p->t_id; //线程编号
    
    for (int k = 0;k < n;++k)
    {
        sem_wait(&sem_workerstart[t_id]);//阻塞，等待主线完成除法操作（操作自己专属的信号量）


        for (int i = k + 1 + t_id; i < n; i += worker_count)
        {
            int j = 0;
            for (j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j]-A[i][k] *A[k][j];
            }

            A[i][k] = 0;
        }
        sem_post(&sem_main);//唤醒主线程
        sem_wait(&sem_workerend[t_id]);//阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
}

int main()
{
    init();

    clock_t  clockBegin, clockEnd;
    clockBegin = clock(); //开始计时


        //初始化信号量
        sem_init(&sem_main, 0, 0);
        for (int i = 0; i < worker_count;++i)
        {
            sem_init(&sem_workerstart[i], 0, 0);
            sem_init(&sem_workerend[i], 0, 0);
        }

        //创建工作线程，进行消去操作

        pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
        threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构

        //分配任务
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            //param[t_id].k = k;
            param[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
        }

        for (int k = 0; k < n; k++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;


            //创建线程
            for (int t_id = 0; t_id < worker_count; t_id++)
                sem_post(&sem_workerstart[t_id]);

            //主线程挂起等待所有的工作线程完成此轮消去工作
            for (int t_id = 0; t_id < worker_count; t_id++)
                sem_wait(&sem_main);

            //主线程再次唤醒工作线程进入下一轮次的消去任务
            for (int t_id = 0; t_id < worker_count;++t_id)
                sem_post(&sem_workerend[t_id]);
        }
        for(int t_id = 0; t_id < worker_count; t_id++)
            pthread_join(handles[t_id], NULL);

        for (int i = 0; i < worker_count; ++i) {
            sem_destroy(&sem_workerstart[i]);
            sem_destroy(&sem_workerend[i]);
        }

    clockEnd = clock();
    cout << "总共耗时： " << clockEnd - clockBegin << "ms" << endl;
    system("pause");
    return 0;
}
