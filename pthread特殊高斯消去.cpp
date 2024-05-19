#include <pthread.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <semaphore.h>
#include <sys/time.h>
# include <arm_neon.h> 
#include <windows.h>
using namespace std;

unsigned int Act[43577][1363] = { 0 };
unsigned int Pas[54274][1363] = { 0 };

const int Num = 1362;
const int pasNum = 54274;
const int lieNum = 43577;

int NUM_THREADS = 7;

sem_t sem_leader;
sem_t* sem_Next = new sem_t[NUM_THREADS - 1];

bool sign;

struct threadParam_t {
	int t_id;
};

void init_A() {

	unsigned int a;
	ifstream infile("消元子.txt");
	char fin[10000] = { 0 };
	int index;

	while (infile.getline(fin, sizeof(fin))) {
		std::stringstream line(fin);
		int biaoji = 0;


		while (line >> a) {
			if (biaoji == 0) {

				index = a;
				biaoji = 1;
			}
			int k = a % 32;
			int j = a / 32;

			int temp = 1 << k;
			Act[index][Num - 1 - j] += temp;
			Act[index][Num] = 1;
		}
	}
}

void init_P() {

	unsigned int a;
	ifstream infile("被消元行.txt");
	char fin[10000] = { 0 };
	int index = 0;

	while (infile.getline(fin, sizeof(fin))) {
		std::stringstream line(fin);
		int biaoji = 0;


		while (line >> a) {
			if (biaoji == 0) {

				Pas[index][Num] = a;
				biaoji = 1;
			}

			int k = a % 32;
			int j = a / 32;

			int temp = 1 << k;
			Pas[index][Num - 1 - j] += temp;
		}
		index++;
	}
}

void* threadFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;
	uint32x4_t va_Pas = vmovq_n_u32(0);
	uint32x4_t va_Act = vmovq_n_u32(0);

	do {

		int i;
		for (i = lieNum - 1; i - 8 >= -1; i -= 8) {

			for (int j = t_id; j < pasNum; j += NUM_THREADS) {

				while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7) {
					int index = Pas[j][Num];

					if (Act[index][Num] == 1) {

						int k;
						for (k = 0; k + 4 <= Num; k += 4) {

							va_Pas = vld1q_u32(&(Pas[j][k]));
							va_Act = vld1q_u32(&(Act[index][k]));

							va_Pas = veorq_u32(va_Pas, va_Act);
							vst1q_u32(&(Pas[j][k]), va_Pas);
						}

						for (; k < Num; k++) {
							Pas[j][k] = Pas[j][k] ^ Act[index][k];
						}

						int num = 0, S_num = 0;
						for (num = 0; num < Num; num++) {
							if (Pas[j][num] != 0) {
								unsigned int temp = Pas[j][num];
								while (temp != 0) {
									temp = temp >> 1;
									S_num++;
								}
								S_num += num * 32;
								break;
							}
						}
						Pas[j][Num] = S_num - 1;
					}
					else {
						break;
					}
				}
			}
		}

		for (i = i + 8; i >= 0; i--) {

			for (int j = t_id; j < pasNum; j += NUM_THREADS) {

				while (Pas[j][Num] == i) {
					if (Act[i][Num] == 1) {

						int k;
						for (k = 0; k + 4 <= Num; k += 4) {

							va_Pas = vld1q_u32(&(Pas[j][k]));
							va_Act = vld1q_u32(&(Act[i][k]));

							va_Pas = veorq_u32(va_Pas, va_Act);
							vst1q_u32(&(Pas[j][k]), va_Pas);
						}

						for (; k < Num; k++) {
							Pas[j][k] = Pas[j][k] ^ Act[i][k];
						}

						int num = 0, S_num = 0;
						for (num = 0; num < Num; num++) {
							if (Pas[j][num] != 0) {
								unsigned int temp = Pas[j][num];
								while (temp != 0) {
									temp = temp >> 1;
									S_num++;
								}
								S_num += num * 32;
								break;
							}
						}
						Pas[j][Num] = S_num - 1;

					}
					else {
						break;
					}
				}
			}
		}
		if (t_id == 0) {
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader);

		}
		else {
			sem_post(&sem_leader);
			sem_wait(&sem_Next[t_id - 1]);
		}

		if (t_id == 0) {


			sign = false;
			for (int i = 0; i < pasNum; i++) {

				int temp = Pas[i][Num];
				if (temp == -1) {

					continue;
				}


				if (Act[temp][Num] == 0) {

					for (int k = 0; k < Num; k++)
						Act[temp][k] = Pas[i][k];

					Pas[i][Num] = -1;

					sign = true;
				}
			}

		}
		if (t_id == 0) {
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Next[i]);
		}

	} while (sign == true);


	pthread_exit(NULL);
}

int main() {
	init_A();
	init_P();

	struct timeval head, tail;
	double seconds;

	gettimeofday(&head, NULL);


	pthread_t* handles = new pthread_t[NUM_THREADS];
	threadParam_t* param = new threadParam_t[NUM_THREADS];

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
	}



	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);


	sem_destroy(&sem_leader);
	sem_destroy(sem_Next);

	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "time: " << seconds << " ms" << endl;

}