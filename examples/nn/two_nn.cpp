#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <math.h>
#include <random>
#include "cuDL/util/math_funcs.hpp"

using namespace std;

typedef struct {
	float *data;
	int row;
	int col;
} DMatrix;

typedef struct{
	DMatrix w1;
	DMatrix w2;
	DMatrix b1;
	DMatrix b2;
} GRADS;

void print_matrix(float *A, int row, int col) {
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			cout << *(A + i * col + j) << ", ";
		}
		cout <<endl;
	}
}

void relu(float *A, const int size) {
	for (int i = 0; i < size; ++i) {
		*(A + i) = max(float(0), *(A + i));
	}
}

void softmax(float *score, int row, int col) {

	for (int i = 0; i < row; ++i) {
		float _sum = 0;

		// compute the exp score, and sum by row
		for (int j = 0; j < col; ++j) {
			*(score + i * col + j) = exp(*(score + i * col + j));
			_sum += *(score + i * col + j);

		}
		
		//normalized by each row
		for (int j = 0; j < col; ++j) {
			*(score + i * col + j) = *(score + i * col + j) / _sum;
		}
		
	}
}

float softmax_loss(float *prob, int row, int col, 
	               const DMatrix y, const DMatrix w1, 
	               const DMatrix w2, float reg) {
	vector<float> corr_probs;
	float loss = 0;

	for (int i = 0; i < y.row; ++i) {
		for (int j = 0; j < y.col; ++j) {
			int label = (int)*(y.data + i * col + j);

			corr_probs.push_back(*(prob + j * col + label));
		}
	}

	for (int i = 0; i < corr_probs.size(); ++i) {
		loss -= log(corr_probs[i]);
	}

	float data_loss = loss / y.col;

	float ssum_w1 = 0;
	float ssum_w2 = 0;

	// compute the regularizaion loss
	for (int i = 0; i < w1.row; ++i) {
		for (int j = 0; j < w1.col; ++j) {
			ssum_w1 += pow(*(w1.data + i * w1.col + j), 2);
		}
	}

	for (int i = 0; i < w2.row; ++i) {
		for (int j = 0; j < w2.col; ++j) {
			ssum_w2 += pow(*(w2.data + i * w2.col + j), 2);
		}
	}

	float reg_loss = reg * (0.5 * ssum_w1 + 0.5 * ssum_w2);

	loss = data_loss + reg_loss;

	return loss;
}

void back_relu(float* dA, float* A, int row, int col){
	for (int i = 0; i < row; ++i) {
		for (int j =0; j < col; ++j) {
			if (*(A + i * col + j) <= 0) {
				*(dA + i * col + j) = 0;
			}
		}
	}
}

vector<int> random_choice(int total, int sample) {
	vector<int> idx;
	for (int i = 0; i < sample; ++i) {
		idx.push_back(rand() % total);
	}

	return idx;
}

void gen_mini_batch(DMatrix x, DMatrix y, float *x_batch, 
	                float *y_batch, vector<int> idx) {
	int x_start = 0;
	int y_start = 0;
	for (int i = 0; i < idx.size(); ++i) {
		memcpy(x_batch + x_start, x.data + idx[i] * x.col, x.col * sizeof(float));
		x_start +=  x.col;

		memcpy(y_batch + y_start, y.data + idx[i], sizeof(float));
		y_start ++;
	}
}




class TwoLayerNets {
	public:
		TwoLayerNets(const int input_size,
					 const int hidden_size,
					 const int number_class,
					 const float& scale);

		void loss(const DMatrix x, 
				  const DMatrix y,
				  float &loss, 
				  GRADS &grads,
				  float reg = 0.0);
		void reg_weight(float *A, const float *B, int row, int col, float reg);

		void train(const DMatrix x, const DMatrix y,
			       const DMatrix x_val, const DMatrix y_val,
			       const float learning_rate=1e-3, 
			       const float learning_rate_decay=0.95,
            	   const float reg=5e-6, const int num_iters=100,
            	   const int batch_size=200, const bool verbose=true);

		void update(DMatrix old_d, DMatrix new_d, float learning_rate);

		void predict();
		~TwoLayerNets();

	private:
		DMatrix w1;
		DMatrix w2;
		DMatrix b1;
		DMatrix b2;
};



TwoLayerNets::TwoLayerNets(const int input_size,
					 	   const int hidden_size,
					       const int number_class,
					       const float& scale) {
	w1.row = input_size;
	w1.col = hidden_size;
	w1.data = (float *)malloc(w1.row * w1.col * sizeof(float));
	memset(w1.data, 0, w1.row * w1.col * sizeof(float));

	default_random_engine generator;
	normal_distribution<float> distribution(0, 1);

	for (int i = 0; i < w1.row; ++i) {
		for (int j = 0; j < w1.col; ++j) {
			*(w1.data + i * w1.col + j) = (float) distribution(generator) * scale;
			// *(w1.data + i * w1.col + j) = w1_test[i][j];
		}
	}

	b1.row = 1;
	b1.col = hidden_size;

	b1.data = (float *)malloc(b1.row * b1.col * sizeof(float));
	memset(b1.data, 0, b1.row * b1.col * sizeof(float));
	for (int i = 0; i < b1.row; ++i) {
		for (int j = 0; j < b1.col; ++j) {
			*(b1.data + i * b1.col + j) = 0;
		}
	}

	w2.row = hidden_size;
	w2.col = number_class;
	w2.data = (float *)malloc(w2.row * w2.col * sizeof(float));
	memset(w2.data, 0, w2.row * w2.col * sizeof(float));

	for (int i = 0; i < w2.row; ++i) {
		for (int j = 0; j < w2.col; ++j) {
			*(w2.data + i * w2.col + j) = (float) distribution(generator) * scale;
			// *(w2.data + i * w2.col + j) = w2_test[i][j];
		}
	}

	b2.row = 1;
	b2.col = number_class;

	b2.data = (float *)malloc(b2.row * b2.col * sizeof(float));
	memset(b2.data, 0, b2.row * b2.col * sizeof(float));
	for (int i = 0; i < b2.row; ++i) {
		for (int j = 0; j < b2.col; ++j) {
			*(b2.data + i * b2.col + j) = 0;
		}
	}
}

TwoLayerNets::~TwoLayerNets() {
	free(w1.data);
	free(w2.data);
	free(b1.data);
	free(b2.data);
}

void TwoLayerNets::reg_weight(float *A, const float *B, int row, int col, float reg) {
	for (int i = 0; i < row * col; ++i) {
		* (A + i) += reg * (* (B + i));
	}
}

void TwoLayerNets::update(DMatrix old_d, DMatrix new_d, float learning_rate){
	for (int i = 0; i < old_d.row * old_d.col; ++i) {
		*(old_d.data + i) += (-1) * learning_rate * (*(new_d.data + i));
	}
}


void TwoLayerNets::loss(const DMatrix x, 
				        const DMatrix y,
				        float &loss, 
				        GRADS &grads,
				        float reg)
{
	/* 1. Perform the forward pass, computing the class scores for the input */
	// Define the hidden layer shape
	float *hidden = (float *)malloc(x.row * w1.col * sizeof(float));
	memset(hidden, 0, x.row * w1.col * sizeof(float));

  	inner_product_cpu(CblasNoTrans, CblasNoTrans, x.row, w1.col , x.col, 
  				  (float)1.0, x.data, w1.data, (float)1.0, hidden);

	// add the bias term
	for (int i = 0; i < x.row; ++i) {
		for (int j = 0; j < w1.col; ++j) {
			*(hidden + i * w1.col + j) += *(b1.data + j);
		}
	}

	// ReLu function for hidden layer
	relu(hidden, x.row * w1.col);

	// Compute the scores for the input
	float *scores = (float *)malloc(x.row * w2.col * sizeof(float));
	memset(scores, 0, x.row * w2.col * sizeof(float));

  	inner_product_cpu(CblasNoTrans, CblasNoTrans, x.row, w2.col, w1.col, 
  				  (float)1.0, hidden, w2.data, (float)1.0, scores);

  	// add the bias term
	for (int i = 0; i < x.row; ++i) {
		for (int j = 0; j < w2.col; ++j) {
			*(scores + i * w2.col + j) += *(b2.data + j);
		}
	}

	// compute the softmax probablity, now scores become probablity
	softmax(scores, x.row, w2.col);

	// compute the softmax loss
	loss = softmax_loss(scores, x.row, w2.col, y, w1, w2, reg);

	/* 2. Backward pass: compute gradients */
	float *dscores = (float *)malloc(x.row * w2.col * sizeof(float));
	memset(dscores, 0, x.row * w2.col * sizeof(float));
	memcpy(dscores, scores, x.row * w2.col * sizeof(float));

	for (int i = 0; i < y.row; ++i) {
		for (int j = 0; j < y.col; ++j) {
			int label = (int)*(y.data + i * y.col + j);

			*(dscores + j * w2.col + label) -= 1;
		}
	}

	for (int i = 0; i < x.row; ++i) {
		for (int j = 0; j < w2.col; ++j) {
			*(dscores + i * w2.col + j) =  *(dscores + i * w2.col + j) / y.col;
		}
	}

	// Compute gridents for w2, and b2
	grads.w2.data = (float*)malloc(w2.row * w2.col * sizeof(float));
	memset(grads.w2.data, 0, w2.row * w2.col * sizeof(float));

	grads.w2.row = w2.row;
	grads.w2.col = w2.col;

	inner_product_cpu(CblasTrans, CblasNoTrans, w1.col, w2.col, x.row, 
	  				  (float)1.0, hidden, dscores, (float)1.0, grads.w2.data);

	// Compute gridents for b2
	grads.b2.data = (float*)malloc(b2.row * b2.col * sizeof(float));
	memset(grads.b2.data, 0, b2.row * b2.col * sizeof(float));
	grads.b2.row = b2.row;
	grads.b2.col = b2.col;

	for (int i = 0; i < x.row; ++i) {
		for (int j = 0; j < w2.col; ++j) {
			*(grads.b2.data + j) += *(dscores + i * w2.col + j);
		}
	}

	// compute deritive for dscores
	float* dhidden = (float*)malloc(x.row * w2.row * sizeof(float));
	memset(dhidden, 0, x.row * w2.row * sizeof(float));
	inner_product_cpu(CblasNoTrans, CblasTrans, x.row, w2.row, w2.col, 
  				  (float)1.0, dscores, w2.data, (float)1.0, dhidden);

	// compute backprop the ReLU non-linearity
	back_relu(dhidden, hidden, x.row, w2.row);

	// compute gridents for w1 and b1
	grads.w1.data = (float*)malloc(w1.row * w1.col * sizeof(float));
	memset(grads.w1.data, 0, w1.row * w1.col * sizeof(float));

	grads.w1.row = w1.row;
	grads.w1.col = w1.col;

	inner_product_cpu(CblasTrans, CblasNoTrans, x.col, w2.row, x.row, 
  				  (float)1.0, x.data, dhidden, (float)1.0, grads.w1.data);

	grads.b1.data = (float*)malloc(b1.row * b1.col * sizeof(float));
	memset(grads.b1.data, 0, b1.row * b1.col * sizeof(float));
	grads.b1.row = b1.row;
	grads.b1.col = b1.col;

	for (int i = 0; i < x.row; ++i) {
		for (int j = 0; j < w2.row; ++j) {
			*(grads.b1.data + j) += *(dhidden + i * w2.row + j);
		}
	}

	reg_weight(grads.w1.data, w1.data, grads.w1.row, grads.w1.col, reg);
	reg_weight(grads.w2.data, w2.data, grads.w2.row, grads.w2.col, reg);

	free(dhidden);
	free(dscores);
  	free(scores);
	free(hidden);
}

void TwoLayerNets::train(const DMatrix x, const DMatrix y,
	       				 const DMatrix x_val, const DMatrix y_val,
	                     const float learning_rate, 
	       			     const float learning_rate_decay,
    	                 const float reg, const int num_iters,
    	                 const int batch_size, const bool verbose) {

	int num_train = x.row;
	int iterations_per_epoch = max(num_train / batch_size, 1);

    vector<float> loss_history;
    vector<float> train_acc_history;
    vector<float> val_acc_history;

	for (int i = 0; i < num_iters; ++i) {
		vector<int> idx = random_choice(num_train, batch_size);

		DMatrix x_batch;
		x_batch.row = batch_size;
		x_batch.col = x.col;
		x_batch.data = (float *)malloc(batch_size * x.col * sizeof(float));
		memset(x_batch.data, 0, batch_size * x.col * sizeof(float));

		DMatrix y_batch;
		y_batch.row = 1;
		y_batch.col = batch_size;

		y_batch.data = (float *)malloc(batch_size * sizeof(float));
		memset(y_batch.data, 0, batch_size * sizeof(float));

		gen_mini_batch(x, y, x_batch.data, y_batch.data, idx);

		float _loss = 0;
		GRADS _grads;

		loss(x_batch, y_batch, _loss, _grads, reg);
		loss_history.push_back(_loss);

		// update weights and biass
		update(w1, _grads.w1, learning_rate);
		update(w2, _grads.w2, learning_rate);
		update(b1, _grads.b1, learning_rate);
		update(b2, _grads.b2, learning_rate);
			
		cout << "The iteration " << i << " / " << num_iters << ": and loss is: " << _loss;
		cout << endl;

		free(_grads.w1.data);
		free(_grads.w2.data);
		free(_grads.b1.data);
		free(_grads.b2.data);
		free(x_batch.data);
		free(y_batch.data);
	}

}

void init_toy_data(float *x, float *y) {
	const int num_inputs = 5;
	const int input_size = 4;

	float input[num_inputs][input_size] = {
		{16.24345364,  -6.11756414,  -5.28171752, -10.72968622},
		{8.65407629, -23.01538697,  17.44811764,  -7.61206901},
		{3.19039096,  -2.49370375,  14.62107937, -20.60140709},
		{-3.22417204,  -3.84054355,  11.33769442, -10.99891267},
		{-1.72428208,  -8.77858418,   0.42213747, 5.82815214},
	};

	float output[num_inputs] = {0, 1, 2, 2, 1};

	memcpy(x, &input[0][0], num_inputs * input_size * sizeof(float));
	memcpy(y, &y[0], num_inputs * sizeof(float));
}

int main(int argc, char** argv) {
	int num_inputs = 5;
	int input_size = 4;
	int hidden_size = 10;
	int num_classes = 3;

	DMatrix x, y;
	x.row = num_inputs;
	x.col = input_size;
	x.data = (float *)malloc(num_inputs * input_size * sizeof(float));
	memset(x.data, 0, num_inputs * input_size * sizeof(float));

	y.row = 1;
	y.col = num_inputs;

	y.data = (float *)malloc(num_inputs * sizeof(float));
	memset(y.data, 0, num_inputs * sizeof(float));

	init_toy_data(x.data, y.data);

	TwoLayerNets *net = new TwoLayerNets(input_size, hidden_size, 
                                 	 	 num_classes, 1e-1);

	net->train(x, y, x, y, 1e-1, 0.95, 5e-6, 100);

	free(x.data);
	free(y.data);
	delete net;
}
