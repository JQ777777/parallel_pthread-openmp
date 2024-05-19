#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <semaphore.h>
#include <sys/time.h>
# include <arm_neon.h> // use Neon
using namespace std;

unsigned int Act[43577][1363] = { 0 };
unsigned int Pas[54274][1363] = { 0 };

const int Num = 1362;
const int pasNum = 54274;
const int lieNum = 43577;

//线程数定义
const int NUM_THREADS = 7; //线程数量

//全局变量定义，用于判断接下来是否进入下一轮
bool sign;

struct threadParam_t
{
    int t_id; // 线程 id
};

//消元子初始化
void init_A()
{
    unsigned int a;
    ifstream infile("消元子.txt");
    char fin[10000] = { 0 };
    int index;
    //从文件中提取行
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

        //从行中提取单个的数字
        while (line >> a)
        {
            if (biaoji == 0)
            {
                //取每行第一个数字为行标
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1;//设置该位置记录消元子该行是否为空，为空则是0，否则为1
        }
    }
}

//被消元行初始化
void init_P()
{
    //直接按照磁盘文件的顺序存，在磁盘文件是第几行，在数组就是第几行
    unsigned int a;
    ifstream infile("被消元行.txt");
    char fin[10000] = { 0 };
    int index = 0;
    //从文件中提取行
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

        //从行中提取单个的数字
        while (line >> a)
        {
            if (biaoji == 0)
            {
                //用Pas[ ][263]存放被消元行每行第一个数字，用于之后的消元操作
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
void f_omp1()
{
    uint32x4_t va_Pas = vmovq_n_u32(0);
    uint32x4_t va_Act = vmovq_n_u32(0);
    bool sign;
#pragma omp parallel num_threads(NUM_THREADS), private(va_Pas, va_Act)
    do
    {
        
        for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
        {
            //每轮处理8个消元子，范围：首项在 i-7 到 i
#pragma omp for schedule(static)
            for (int j = 0; j < pasNum; j++)
            {
                //看被消元行有没有首项在此范围内的
                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
                {
                    int index = Pas[j][Num];

                    if (Act[index][Num] == 1)//消元子不为空
                    {
                        int k;
                        for (k = 0; k + 4 <= Num; k += 4)
                        {
                            //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                            va_Pas = vld1q_u32(&(Pas[j][k]));
                            va_Act = vld1q_u32(&(Act[index][k]));

                            va_Pas = veorq_u32(va_Pas, va_Act);
                            vst1q_u32(&(Pas[j][k]), va_Pas);
                        }

                        for (; k < Num; k++)
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        }
                       
                        //更新Pas[j][18]存的首项值
                        //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                        //找异或之后Pas[j][ ]的首项
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++)
                        {
                            if (Pas[j][num] != 0)
                            {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0)
                                {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;
                    }
                    else//消元子为空
                    {
                        break;
                    }
                }
            }
        }

        for (int i = lieNum % 8 - 1; i >= 0; i--)
        {
            //每轮处理1个消元子，范围：首项等于i
#pragma omp for schedule(static)
            for (int j = 0; j < pasNum; j++)
            {
                //看53个被消元行有没有首项等于i的
                while (Pas[j][Num] == i)
                {
                    if (Act[i][Num] == 1)//消元子不为空
                    {
                       
                        int k;
                        for (k = 0; k + 4 <= Num; k += 4)
                        {
                            //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                            va_Pas = vld1q_u32(&(Pas[j][k]));
                            va_Act = vld1q_u32(&(Act[i][k]));

                            va_Pas = veorq_u32(va_Pas, va_Act);
                            vst1q_u32(&(Pas[j][k]), va_Pas);
                        }

                        for (; k < Num; k++)
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        }
                       
                        //更新Pas[j][18]存的首项值
                        //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                        //找异或之后Pas[j][ ]的首项
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++)
                        {
                            if (Pas[j][num] != 0)
                            {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0)
                                {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;

                    }
                    else//消元子为空
                    {
                        break;
                    }
                }
            }
        }

    //升格消元子，然后判断是否结束
#pragma omp single
        {
            sign = false;
            for (int i = 0; i < pasNum; i++)
            {
                //找到第i个被消元行的首项
                int temp = Pas[i][Num];
                if (temp == -1)
                {
                    //说明他已经被升格为消元子了
                    continue;
                }

                //看这个首项对应的消元子是不是为空，若为空，则补齐
                if (Act[temp][Num] == 0)
                {
                    //补齐消元子
                    for (int k = 0; k < Num; k++)
                        Act[temp][k] = Pas[i][k];
                    //将被消元行升格
                    Pas[i][Num] = -1;
                    //标志bool设为true，说明此轮还需继续
                    sign = true;
                }
            }
        }

    } while (sign == true);
}

void f_omp2()
{
    //uint32x4_t va_Pas =  vmovq_n_u32(0);
    //uint32x4_t va_Act =  vmovq_n_u32(0);
#pragma omp parallel num_threads(NUM_THREADS)
//, private(va_Pas, va_Act)
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {

#pragma omp for schedule(static)
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    /*
                    for (int k = 0; k <= Num-4; k+=4)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        va_Pas =  vld1q_u32(& (Pas[j][k]));
                        va_Act =  vld1q_u32(& (Act[index][k]));

                        va_Pas = veorq_u32(va_Pas,va_Act);
                        vst1q_u32( &(Pas[j][k]) , va_Pas );
                    }

                    for(int k=Num-Num%4 ; k<Num; k++ )
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    */
                    
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }

    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {

                    
#pragma omp for schedule(static)
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }

                    /*
                    for (int k = 0; k <= Num-4; k+=4)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        va_Pas =  vld1q_u32(& (Pas[j][k]));
                        va_Act =  vld1q_u32(& (Act[i][k]));

                        va_Pas = veorq_u32(va_Pas,va_Act);
                        vst1q_u32( &(Pas[j][k]) , va_Pas );
                    }

                    for(int k=Num-Num%4; k<Num; k++ )
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    */
                   
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}

void f_ordinary()
{
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        //每轮处理8个消元子，范围：首项在 i-7 到 i

        for (int j = 0; j < pasNum; j++)
        {
            //看4535个被消元行有没有首项在此范围内的
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)//消元子不为空
                {
                    //Pas[j][]和Act[（Pas[j][18]）][]做异或
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }

                    //更新Pas[j][18]存的首项值
                    //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                    //找异或之后Pas[j][ ]的首项
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else//消元子为空
                {
                    //Pas[j][]来补齐消元子
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;//设置消元子非空
                    break;
                }

            }
        }
    }

    for (i = i + 8; i >= 0; i--)
    {
        //每轮处理1个消元子，范围：首项等于i

        for (int j = 0; j < pasNum; j++)
        {
            //看53个被消元行有没有首项等于i的
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)//消元子不为空
                {
                    //Pas[j][]和Act[i][]做异或
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }

                    //更新Pas[j][18]存的首项值
                    //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                    //找异或之后Pas[j][ ]的首项
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else//消元子为空
                {
                    //Pas[j][]来补齐消元子
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;//设置消元子非空
                    break;
                }
            }
        }
    }
}

int main()
{
    init_A();
    init_P();

    // 计时操作
    struct timeval head, tail;
    double seconds;

    gettimeofday(&head, NULL);//开始计时

    f_omp2();

    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout << "time: " << seconds << " ms" << endl;
    return 0;
}