#include <omp.h>
#include <iostream>
using namespace std;

const int n = 1000;
float arr[n][n];
float A[n][n];
const int NUM_THREADS = 2; //工作线程数量


void init()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			arr[i][j] = 0;
		}
		arr[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			arr[i][j] = rand() % 100;
	}

	for (int i = 0; i < n; i++)
	{
		int k1 = rand() % n;
		int k2 = rand() % n;
		for (int j = 0; j < n; j++)
		{
			arr[i][j] += arr[0][j];
			arr[k1][j] += arr[k2][j];
		}
	}
}


void ReStart()
{
	for (int i = 0;i < n;i++)
	{
		for (int j = 0;j < n;j++)
			A[i][j] = arr[i][j];
	}
}

void f_omp_guided()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(guided, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
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

void f_omp_static()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}


void f_omp_dynamic()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(dynamic, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}



int main()
{
	init();

	// 使用普通方法进行计算
	ReStart();
	clock_t  clockBegin1, clockEnd1;
	clockBegin1 = clock(); //开始计时
	f_ordinary();
	clockEnd1 = clock();
	cout << "f_ordinary: " << clockEnd1 - clockBegin1 << "ms" << endl;

	// 使用静态调度的 OpenMP 方法进行计算
	ReStart();
	clock_t  clockBegin2, clockEnd2;
	clockBegin2 = clock(); //开始计时
	f_omp_static();
	clockEnd2 = clock();
	cout << "f_omp_static: " << clockEnd2 - clockBegin2 << "ms" << endl;

	// 使用动态调度的 OpenMP 方法进行计算
	ReStart();
	clock_t  clockBegin3, clockEnd3;
	clockBegin3 = clock(); //开始计时
	f_omp_dynamic();
	clockEnd3 = clock();
	cout << "f_omp_dynamic： " << clockEnd3 - clockBegin3 << "ms" << endl;

	// 使用向导调度的 OpenMP 方法进行计算
	ReStart();
	clock_t  clockBegin4, clockEnd4;
	clockBegin4 = clock(); //开始计时
	f_omp_guided();
	clockEnd4 = clock();
	cout << "f_omp_guided: " << clockEnd4 - clockBegin4 << "ms" << endl;

	return 0;
}
