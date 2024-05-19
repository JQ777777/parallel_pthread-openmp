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

const int n = 400;
float A[n][n];
int NUM_THREADS = 2;

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

//信号量定义
sem_t sem_leader;
sem_t* sem_Divsion = new sem_t[NUM_THREADS - 1]; // 每个线程有自己专属的信号量
sem_t* sem_Elimination = new sem_t[NUM_THREADS - 1];



//线程函数定义（穿插）
void* threadFunc_horizontal1(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{
		if (t_id == 0)
		{
			// 假设A是一个二维float数组，k是行索引，n是列数  
			float divisor = A[k][k];


			// 计算除数的倒数  
			float reciprocal = 1.0f / divisor;

			// 创建一个包含除数倒数的AVX向量  
			__m256 reciprocal_vec = _mm256_set1_ps(reciprocal);

			// 处理A[k]行的剩余部分（从k+1列开始）  
			int j;
			for (j = k + 1; j < n - 7; j += 8) // 确保不会越界，并且每次增加8  
			{
				// 加载A[k][j]到AVX向量  
				__m256 Akj_vec = _mm256_loadu_ps(&A[k][j]);

				// 计算归一化后的值  
				__m256 normalized_vec = _mm256_mul_ps(Akj_vec, reciprocal_vec);

				// 将结果存储回A[k][j]  
				_mm256_storeu_ps(&A[k][j], normalized_vec);
			}

			// 处理剩余的少于8个元素的列（如果有的话）  
			for (; j < n; j++)
			{
				A[k][j] *= reciprocal; // 直接使用标量乘法  
			}

			// 设置A[k][k]为1.0（这是标量操作，因为只有一个元素）  
			A[k][k] = 1.0f;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}

		//循环划分任务
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			float Aik = A[i][k];

			// 创建一个包含-Aik的AVX向量（用于后续的乘法操作）  
			__m256 Aik_vec = _mm256_set1_ps(-Aik);

			// 消去过程  
			for (int j = k + 1; j < n - 7; j += 8) // 确保不会越界，并且每次增加8  
			{
				// 加载A[k][j]到AVX向量  
				__m256 Akj_vec = _mm256_loadu_ps(&A[k][j]);

				// 加载A[i][j]到AVX向量  
				__m256 Aij_vec = _mm256_loadu_ps(&A[i][j]);

				// 计算A[i][j] = A[i][j] - Aik * A[k][j]的向量形式  
				__m256 result_vec = _mm256_add_ps(Aij_vec, _mm256_mul_ps(Aik_vec, Akj_vec));

				// 存储结果到A[i][j]  
				_mm256_storeu_ps(&A[i][j], result_vec);
			}

			// 处理剩余的少于8个元素的列（如果有的话）  
			for (int j = n - 7; j < n; j++)
			{
				A[i][j] = A[i][j] - Aik * A[k][j]; // 使用标量操作  
			}

			// 设置A[i][k]为0（这是标量操作，因为只有一个元素）  
			A[i][k] = 0.0f;
		}

		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // 等待其它 worker 完成消去

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
}

int main()
{
	init();

	clock_t  clockBegin, clockEnd;
	clockBegin = clock(); //开始计时

	//初始化信号量
	sem_init(&sem_leader, 0, 0);

	for (int i = 0; i < NUM_THREADS - 1; ++i)
	{
		sem_init(sem_Divsion, 0, 0);
		sem_init(sem_Elimination, 0, 0);
	}

	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
	threadParam_t* param = new threadParam_t[NUM_THREADS];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc_horizontal1, (void*)&param[t_id]);
	}


	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//销毁所有信号量
	sem_destroy(&sem_leader);
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);


	clockEnd = clock();
	cout << "总共耗时： " << clockEnd - clockBegin << "ms" << endl;
	system("pause");
	return 0;
}