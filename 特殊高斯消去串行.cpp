#include <iostream>
#include <sstream>
#include <fstream>
#include <windows.h>
using namespace std;

const int Num = 1000;
const int pasNum = 10000;
const int lieNum = 30000;

unsigned int Act[lieNum][Num] = { 0 };
unsigned int Pas[lieNum][Num] = { 0 };


void init_A()
{
    unsigned int a;
    ifstream infile("消元子.txt");
    char fin[10000] = { 0 };
    int index;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;
        while (line >> a)
        {
            if (biaoji == 0)
            {
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
void init_P()
{
    unsigned int a;
    ifstream infile("被消元行.txt");
    char fin[10000] = { 0 };
    int index = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;
        while (line >> a)
        {
            if (biaoji == 0)
            {
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




void f_ordinary(){
    bool sign;
    do{
        for (int i = lieNum - 1; i - 8 >= -1; i -= 8){
            for (int j = 0; j < pasNum; j++){

                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7){
                    int index = Pas[j][Num];
                    if (Act[index][Num] == 1){
                        for (int k = 0; k < Num; k++){
                            Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        }
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++){
                            if (Pas[j][num] != 0){
                                unsigned int temp = Pas[j][num];
                                while (temp != 0){
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;
                    }
                    else{
                        break;
                    }
                }
            }
        }
        for (int i = lieNum % 8 - 1; i >= 0; i--){
            for (int j = 0; j < pasNum; j++){
                while (Pas[j][Num] == i){
                    if (Act[i][Num] == 1){
                        for (int k = 0; k < Num; k++){
                            Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        }
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++){
                            if (Pas[j][num] != 0){
                                unsigned int temp = Pas[j][num];
                                while (temp != 0){
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;
                    }
                    else{
                        break;
                    }
                }
            }
        }
        sign = false;
        for (int i = 0; i < pasNum; i++){
            int temp = Pas[i][Num];
            if (temp == -1){
                continue;
            }
            if (Act[temp][Num] == 0){
                for (int k = 0; k < Num; k++)
                    Act[temp][k] = Pas[i][k];
                Pas[i][Num] = -1;
                sign = true;
            }
        }
    } while (sign == true);
}

int main()
{
    double seconds;
    long long head, tail, freq, noww;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    init_A();
    init_P();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f_ordinary();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;
    cout << seconds;
    cout << "ms";
}