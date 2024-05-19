#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h> //AVX、AVX2
using namespace std;

const int n = 1000;
float A[n][n];
int NUM_THREADS = 7;

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
	int t_id; //线程 id
};

//barrier 定义
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;


//线程函数定义
void* threadFunc(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{
		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j< n; j++)
			{
				A[k][j] = A[k][j] / A[k][k];
			}
			A[k][k] = 1.0;
		}

		//第一个同步点
		pthread_barrier_wait(&barrier_Divsion);

		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			//消去
			int j;
			for (j = k + 1; j <= n; j ++)
			{
				A[i][j] = A[i][j]-A[i][k] *A[k][j];
			}
			A[i][k] = 0;
		}
		// 第二个同步点
		pthread_barrier_wait(&barrier_Elimination);

	}
	pthread_exit(NULL);
}

int main()
{
	init();

	clock_t  clockBegin, clockEnd;
	clockBegin = clock(); //开始计时

	//初始化barrier
	pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);


	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
	threadParam_t* param = new threadParam_t[NUM_THREADS];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
	}


	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//销毁所有的 barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);

	clockEnd = clock();
	cout << "总共耗时： " << clockEnd - clockBegin << "ms" << endl;
	system("pause");
	return 0;
}