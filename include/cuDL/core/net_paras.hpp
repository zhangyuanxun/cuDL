#ifndef NET_PARAS_HPP_
#define NET_PARAS_HPP_

#include <string>

using namespace std;

const int CPU_MODE = 0;
const int GPU_MODE = 1;

typedef struct {
	string name;
	float learning_rate;
	float reg;
	int max_iter;
	int mode;
} NetParas;

#endif