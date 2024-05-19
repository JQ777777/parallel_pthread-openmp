#include <omp.h>
#include <iostream>
#include <emmintrin.h>
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
		for (int i = k + 1; i < n; i += 4) // 每次处理4行  
		{
			// 加载tmp值到一个向量中（如果i不是4的倍数，则需要特殊处理）  
			__m128 tmp_vec = _mm_setzero_ps(); // 初始化一个零向量  
			if (i < n - 3) { // 确保不会越界  
				tmp_vec = _mm_loadu_ps(&A[i][k]); // 加载4个float值，不需要对齐  
			}
			else {
				// 如果i接近n的末尾，则需要构造一个只包含有效tmp值的向量  
				// 这里简化为只处理前几个元素，其余置零  
				float tmp_array[4] = { 0, 0, 0, 0 };
				if (i < n) tmp_array[0] = A[i][k];
				if (i + 1 < n) tmp_array[1] = A[i + 1][k];
				if (i + 2 < n) tmp_array[2] = A[i + 2][k];
				tmp_vec = _mm_load_ps(tmp_array); // 假设tmp_array是16字节对齐的  
			}

			// 将tmp_vec广播到一个4x4的矩阵中，用于后续的乘法  
			__m128 broadcast_tmp = _mm_shuffle_ps(tmp_vec, tmp_vec, _MM_SHUFFLE(0, 0, 0, 0));

			for (int j = k + 1; j < n; j += 4) // 每次处理4列  
			{
				// 加载A[k][j]的4个float值到一个向量中  
				__m128 A_kj_vec = _mm_loadu_ps(&A[k][j]); // 不需要对齐  

				// 计算tmp * A[k][j]的4个值  
				__m128 mult_vec = _mm_mul_ps(broadcast_tmp, A_kj_vec);

				// 加载A[i][j]的4个float值到一个向量中（可能需要处理边界情况）  
				__m128 A_ij_vec = _mm_setzero_ps(); // 初始化一个零向量  
				if (i < n - 3) { // 确保不会越界  
					A_ij_vec = _mm_loadu_ps(&A[i][j]); // 加载4个float值，不需要对齐  
				}
				else {
					// 如果i接近n的末尾，则需要构造一个只包含有效A[i][j]值的向量  
					// ...（类似tmp_vec的构造）  
				}

				// 从A[i][j]中减去tmp * A[k][j]  
				__m128 result_vec = _mm_sub_ps(A_ij_vec, mult_vec);

				// 将结果存回A[i][j]  
				_mm_storeu_ps(&A[i][j], result_vec);
			}

			// 将A[i][k]设置为0（需要特殊处理，因为SSE不直接支持单个元素的设置）  
			for (int idx = 0; idx < 4 && i + idx < n; ++idx) {
				A[i + idx][k] = 0.0f;
			}
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
		for (int i = k + 1; i < n; i += 4) // 每次处理4行  
		{
			// 加载tmp值到一个向量中（如果i不是4的倍数，则需要特殊处理）  
			__m128 tmp_vec = _mm_setzero_ps(); // 初始化一个零向量  
			if (i < n - 3) { // 确保不会越界  
				tmp_vec = _mm_loadu_ps(&A[i][k]); // 加载4个float值，不需要对齐  
			}
			else {
				// 如果i接近n的末尾，则需要构造一个只包含有效tmp值的向量  
				// 这里简化为只处理前几个元素，其余置零  
				float tmp_array[4] = { 0, 0, 0, 0 };
				if (i < n) tmp_array[0] = A[i][k];
				if (i + 1 < n) tmp_array[1] = A[i + 1][k];
				if (i + 2 < n) tmp_array[2] = A[i + 2][k];
				tmp_vec = _mm_load_ps(tmp_array); // 假设tmp_array是16字节对齐的  
			}

			// 将tmp_vec广播到一个4x4的矩阵中，用于后续的乘法  
			__m128 broadcast_tmp = _mm_shuffle_ps(tmp_vec, tmp_vec, _MM_SHUFFLE(0, 0, 0, 0));

			for (int j = k + 1; j < n; j += 4) // 每次处理4列  
			{
				// 加载A[k][j]的4个float值到一个向量中  
				__m128 A_kj_vec = _mm_loadu_ps(&A[k][j]); // 不需要对齐  

				// 计算tmp * A[k][j]的4个值  
				__m128 mult_vec = _mm_mul_ps(broadcast_tmp, A_kj_vec);

				// 加载A[i][j]的4个float值到一个向量中（可能需要处理边界情况）  
				__m128 A_ij_vec = _mm_setzero_ps(); // 初始化一个零向量  
				if (i < n - 3) { // 确保不会越界  
					A_ij_vec = _mm_loadu_ps(&A[i][j]); // 加载4个float值，不需要对齐  
				}
				else {
					// 如果i接近n的末尾，则需要构造一个只包含有效A[i][j]值的向量  
					// ...（类似tmp_vec的构造）  
				}

				// 从A[i][j]中减去tmp * A[k][j]  
				__m128 result_vec = _mm_sub_ps(A_ij_vec, mult_vec);

				// 将结果存回A[i][j]  
				_mm_storeu_ps(&A[i][j], result_vec);
			}

			// 将A[i][k]设置为0（需要特殊处理，因为SSE不直接支持单个元素的设置）  
			for (int idx = 0; idx < 4 && i + idx < n; ++idx) {
				A[i + idx][k] = 0.0f;
			}
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
		for (int i = k + 1; i < n; i += 4) // 每次处理4行  
		{
			// 加载tmp值到一个向量中（如果i不是4的倍数，则需要特殊处理）  
			__m128 tmp_vec = _mm_setzero_ps(); // 初始化一个零向量  
			if (i < n - 3) { // 确保不会越界  
				tmp_vec = _mm_loadu_ps(&A[i][k]); // 加载4个float值，不需要对齐  
			}
			else {
				// 如果i接近n的末尾，则需要构造一个只包含有效tmp值的向量  
				// 这里简化为只处理前几个元素，其余置零  
				float tmp_array[4] = { 0, 0, 0, 0 };
				if (i < n) tmp_array[0] = A[i][k];
				if (i + 1 < n) tmp_array[1] = A[i + 1][k];
				if (i + 2 < n) tmp_array[2] = A[i + 2][k];
				tmp_vec = _mm_load_ps(tmp_array); // 假设tmp_array是16字节对齐的  
			}

			// 将tmp_vec广播到一个4x4的矩阵中，用于后续的乘法  
			__m128 broadcast_tmp = _mm_shuffle_ps(tmp_vec, tmp_vec, _MM_SHUFFLE(0, 0, 0, 0));

			for (int j = k + 1; j < n; j += 4) // 每次处理4列  
			{
				// 加载A[k][j]的4个float值到一个向量中  
				__m128 A_kj_vec = _mm_loadu_ps(&A[k][j]); // 不需要对齐  

				// 计算tmp * A[k][j]的4个值  
				__m128 mult_vec = _mm_mul_ps(broadcast_tmp, A_kj_vec);

				// 加载A[i][j]的4个float值到一个向量中（可能需要处理边界情况）  
				__m128 A_ij_vec = _mm_setzero_ps(); // 初始化一个零向量  
				if (i < n - 3) { // 确保不会越界  
					A_ij_vec = _mm_loadu_ps(&A[i][j]); // 加载4个float值，不需要对齐  
				}
				else {
					// 如果i接近n的末尾，则需要构造一个只包含有效A[i][j]值的向量  
					// ...（类似tmp_vec的构造）  
				}

				// 从A[i][j]中减去tmp * A[k][j]  
				__m128 result_vec = _mm_sub_ps(A_ij_vec, mult_vec);

				// 将结果存回A[i][j]  
				_mm_storeu_ps(&A[i][j], result_vec);
			}

			// 将A[i][k]设置为0（需要特殊处理，因为SSE不直接支持单个元素的设置）  
			for (int idx = 0; idx < 4 && i + idx < n; ++idx) {
				A[i + idx][k] = 0.0f;
			}
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
