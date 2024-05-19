#include <omp.h>
#include <iostream>
#include <immintrin.h>
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
		for (int i = k + 1; i < n; i += 8) // 每次处理8行，因为AVX使用256位，即8个float  
		{
			__m256 tmp_vec = _mm256_setzero_ps(); // 初始化一个零向量  

			if (i < n - 7) // 确保不会越界（因为我们需要加载8个float）  
			{
				tmp_vec = _mm256_loadu_ps(&A[i][k]); // 加载8个float值，不需要对齐  
			}
			else
			{
				// 如果i接近n的末尾，则需要构造一个只包含有效tmp值的向量  
				// 这里简化为只处理前几个元素，其余置零  
				float tmp_array[8] = { 0 };
				for (int idx = 0; idx < 8 && i + idx < n; ++idx)
				{
					tmp_array[idx] = A[i + idx][k];
				}
				tmp_vec = _mm256_loadu_ps(tmp_array); // 不需要tmp_array是32字节对齐的，因为使用了_mm256_loadu_ps  
			}

			// 将tmp_vec广播到一个8x1的向量中，用于后续的乘法  
			// AVX没有直接的广播指令，但我们可以使用permute和shuffle的组合  
			__m256 broadcast_tmp = _mm256_permute_ps(tmp_vec, _MM_SHUFFLE(0, 0, 0, 0)); // 复制第一个float到所有位置  
			broadcast_tmp = _mm256_shuffle_ps(broadcast_tmp, broadcast_tmp, _MM_SHUFFLE(0, 0, 0, 0)); // 再次广播以填充所有位置  

			for (int j = k + 1; j < n; j += 8) // 每次处理8列  
			{
				__m256 A_kj_vec = _mm256_setzero_ps();
				if (j < n - 7) // 确保不会越界  
				{
					A_kj_vec = _mm256_loadu_ps(&A[k][j]); // 加载8个float值，不需要对齐  
				}
				else
				{
					// 如果j接近n的末尾，则需要构造一个只包含有效A[k][j]值的向量  
					// ...（类似tmp_vec的构造）  
				}

				// 计算tmp * A[k][j]的8个值  
				__m256 mult_vec = _mm256_mul_ps(broadcast_tmp, A_kj_vec);

				__m256 A_ij_vec = _mm256_setzero_ps();
				if (i < n - 7) // 确保不会越界  
				{
					A_ij_vec = _mm256_loadu_ps(&A[i][j]); // 加载8个float值，不需要对齐  
				}
				else
				{
					// 如果i接近n的末尾，则需要构造一个只包含有效A[i][j]值的向量  
					// ...（类似tmp_vec的构造）  
				}

				// 从A[i][j]中减去tmp * A[k][j]  
				__m256 result_vec = _mm256_sub_ps(A_ij_vec, mult_vec);

				// 将结果存回A[i][j]  
				_mm256_storeu_ps(&A[i][j], result_vec);
			}

			// 将A[i][k]设置为0（需要特殊处理，因为AVX不直接支持单个元素的设置）  
			for (int idx = 0; idx < 8 && i + idx < n; ++idx)
			{
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
		for (int i = k + 1; i < n; i += 8) // 每次处理8行，因为AVX使用256位，即8个float  
		{
			__m256 tmp_vec = _mm256_setzero_ps(); // 初始化一个零向量  

			if (i < n - 7) // 确保不会越界（因为我们需要加载8个float）  
			{
				tmp_vec = _mm256_loadu_ps(&A[i][k]); // 加载8个float值，不需要对齐  
			}
			else
			{
				// 如果i接近n的末尾，则需要构造一个只包含有效tmp值的向量  
				// 这里简化为只处理前几个元素，其余置零  
				float tmp_array[8] = { 0 };
				for (int idx = 0; idx < 8 && i + idx < n; ++idx)
				{
					tmp_array[idx] = A[i + idx][k];
				}
				tmp_vec = _mm256_loadu_ps(tmp_array); // 不需要tmp_array是32字节对齐的，因为使用了_mm256_loadu_ps  
			}

			// 将tmp_vec广播到一个8x1的向量中，用于后续的乘法  
			// AVX没有直接的广播指令，但我们可以使用permute和shuffle的组合  
			__m256 broadcast_tmp = _mm256_permute_ps(tmp_vec, _MM_SHUFFLE(0, 0, 0, 0)); // 复制第一个float到所有位置  
			broadcast_tmp = _mm256_shuffle_ps(broadcast_tmp, broadcast_tmp, _MM_SHUFFLE(0, 0, 0, 0)); // 再次广播以填充所有位置  

			for (int j = k + 1; j < n; j += 8) // 每次处理8列  
			{
				__m256 A_kj_vec = _mm256_setzero_ps();
				if (j < n - 7) // 确保不会越界  
				{
					A_kj_vec = _mm256_loadu_ps(&A[k][j]); // 加载8个float值，不需要对齐  
				}
				else
				{
					// 如果j接近n的末尾，则需要构造一个只包含有效A[k][j]值的向量  
					// ...（类似tmp_vec的构造）  
				}

				// 计算tmp * A[k][j]的8个值  
				__m256 mult_vec = _mm256_mul_ps(broadcast_tmp, A_kj_vec);

				__m256 A_ij_vec = _mm256_setzero_ps();
				if (i < n - 7) // 确保不会越界  
				{
					A_ij_vec = _mm256_loadu_ps(&A[i][j]); // 加载8个float值，不需要对齐  
				}
				else
				{
					// 如果i接近n的末尾，则需要构造一个只包含有效A[i][j]值的向量  
					// ...（类似tmp_vec的构造）  
				}

				// 从A[i][j]中减去tmp * A[k][j]  
				__m256 result_vec = _mm256_sub_ps(A_ij_vec, mult_vec);

				// 将结果存回A[i][j]  
				_mm256_storeu_ps(&A[i][j], result_vec);
			}

			// 将A[i][k]设置为0（需要特殊处理，因为AVX不直接支持单个元素的设置）  
			for (int idx = 0; idx < 8 && i + idx < n; ++idx)
			{
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
		for (int i = k + 1; i < n; i += 8) // 每次处理8行，因为AVX使用256位，即8个float  
		{
			__m256 tmp_vec = _mm256_setzero_ps(); // 初始化一个零向量  

			if (i < n - 7) // 确保不会越界（因为我们需要加载8个float）  
			{
				tmp_vec = _mm256_loadu_ps(&A[i][k]); // 加载8个float值，不需要对齐  
			}
			else
			{
				// 如果i接近n的末尾，则需要构造一个只包含有效tmp值的向量  
				// 这里简化为只处理前几个元素，其余置零  
				float tmp_array[8] = { 0 };
				for (int idx = 0; idx < 8 && i + idx < n; ++idx)
				{
					tmp_array[idx] = A[i + idx][k];
				}
				tmp_vec = _mm256_loadu_ps(tmp_array); // 不需要tmp_array是32字节对齐的，因为使用了_mm256_loadu_ps  
			}

			// 将tmp_vec广播到一个8x1的向量中，用于后续的乘法  
			// AVX没有直接的广播指令，但我们可以使用permute和shuffle的组合  
			__m256 broadcast_tmp = _mm256_permute_ps(tmp_vec, _MM_SHUFFLE(0, 0, 0, 0)); // 复制第一个float到所有位置  
			broadcast_tmp = _mm256_shuffle_ps(broadcast_tmp, broadcast_tmp, _MM_SHUFFLE(0, 0, 0, 0)); // 再次广播以填充所有位置  

			for (int j = k + 1; j < n; j += 8) // 每次处理8列  
			{
				__m256 A_kj_vec = _mm256_setzero_ps();
				if (j < n - 7) // 确保不会越界  
				{
					A_kj_vec = _mm256_loadu_ps(&A[k][j]); // 加载8个float值，不需要对齐  
				}
				else
				{
					// 如果j接近n的末尾，则需要构造一个只包含有效A[k][j]值的向量  
					// ...（类似tmp_vec的构造）  
				}

				// 计算tmp * A[k][j]的8个值  
				__m256 mult_vec = _mm256_mul_ps(broadcast_tmp, A_kj_vec);

				__m256 A_ij_vec = _mm256_setzero_ps();
				if (i < n - 7) // 确保不会越界  
				{
					A_ij_vec = _mm256_loadu_ps(&A[i][j]); // 加载8个float值，不需要对齐  
				}
				else
				{
					// 如果i接近n的末尾，则需要构造一个只包含有效A[i][j]值的向量  
					// ...（类似tmp_vec的构造）  
				}

				// 从A[i][j]中减去tmp * A[k][j]  
				__m256 result_vec = _mm256_sub_ps(A_ij_vec, mult_vec);

				// 将结果存回A[i][j]  
				_mm256_storeu_ps(&A[i][j], result_vec);
			}

			// 将A[i][k]设置为0（需要特殊处理，因为AVX不直接支持单个元素的设置）  
			for (int idx = 0; idx < 8 && i + idx < n; ++idx)
			{
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
