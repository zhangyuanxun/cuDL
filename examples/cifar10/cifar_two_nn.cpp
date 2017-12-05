#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <math.h>
#include <random>
#include "cifar10_reader.hpp"
#include <numeric>
#include <time.h>
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
			// cout << "(" << *(score + i * col + j) << ", " << exp(*(score + i * col + j)) << ")" << endl;
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
					 const float& scale=1e-4);

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

		vector<float> predict(DMatrix x);
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

  	inner_product_cpu(CblasNoTrans, CblasNoTrans, x.row, w1.col, x.col, 
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

int find_max_idx(vector<float> A) {
	float max_val = INT_MIN;
	int max_idx = 0;

	for (int i = 0; i < A.size(); ++i) {
		if (A[i] > max_val) {
			max_idx = i;
			max_val = A[i];
		}
	}

	return max_idx;
}

vector<float> TwoLayerNets::predict(DMatrix x) {

	vector<float> pred_labels;
	float *hidden = (float *)malloc(x.row * w1.col * sizeof(float));
	memset(hidden, 0, x.row * w1.col * sizeof(float));

  	inner_product_cpu(CblasNoTrans, CblasNoTrans, x.row, w1.col, x.col, 
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

	for (int i = 0; i < x.row; ++i) {
		float max_val = INT_MIN;
		float idx = 0;

		for (int j = 0; j < w2.col; ++j) {

			if (*(scores + i * w2.col + j) > max_val) {
				idx = j;
				max_val = *(scores + i * w2.col + j);
			}
		}

		pred_labels.push_back(idx);
	}

	return pred_labels;
}

void vector_addition(vector<float> &A, vector<float> B) {
	for (int i = 0; i < A.size(); ++i){
		A[i] += B[i];
	}
}


void load_cifar10(float *x, float *y, 
	              float *x_val, float *y_val, 
	              float *x_test, float *y_test,
	              int num_training=49000,
	              int num_validation=1000,
	              int num_test=1000) {
	auto dataset = read_dataset<vector, vector, uint8_t, uint8_t>();	
	vector <vector <float> > training_images;
	vector <vector <float> > test_images;

	vector<float> image_mean (dataset.training_images[0].size(), 0);


	for (int i = 0; i < dataset.training_images.size(); ++i) {
		
		vector <float> value_per_image(dataset.training_images[i].begin(),
			                           dataset.training_images[i].end());
		vector_addition(image_mean, value_per_image);
		training_images.push_back(value_per_image);
	}

	vector <float> training_labels(dataset.training_labels.begin(),
			        			   dataset.training_labels.end());

	for (int i = 0; i < dataset.test_images.size(); ++i) {
		
		vector <float> value_per_image(dataset.test_images[i].begin(),
			                           dataset.test_images[i].end());

		vector_addition(image_mean, value_per_image);
		test_images.push_back(value_per_image);
	}

	vector <float> test_labels(dataset.test_labels.begin(),
			        		   dataset.test_labels.end());

	// compute the mean of image
	for (int i = 0; i < image_mean.size(); ++i) {
		image_mean[i] = image_mean[i] / (training_images.size() + test_images.size());
	}

	// Normalize the data: subtract the mean image
	for (int i = 0; i < training_images.size(); ++i) {
		for (int j = 0; j < training_images[i].size(); ++j) {
			training_images[i][j] -= image_mean[j];
		}
	}

	for (int i = 0; i < test_images.size(); ++i) {
		for (int j = 0; j < test_images[i].size(); ++j) {
			test_images[i][j] -= image_mean[j];
		}
	}


	//subsampling traning and validation
	vector<vector<float> > training_x(training_images.begin(), training_images.begin() + num_training);
	for (int i = 0; i < training_x.size(); ++i){
		for (int j = 0; j < training_x[0].size(); ++j) {
			*(x + i * training_x[0].size() + j) = training_x[i][j];
		}
	}

	vector<float> training_y(training_labels.begin(), training_labels.begin() + num_training);
	for (int i = 0; i < training_y.size(); ++i){
		*(y + i) = training_y[i];
	}

	vector<vector<float> > validation_x(training_images.begin() + num_training, training_images.end());
	for (int i = 0; i < validation_x.size(); ++i){
		for (int j = 0; j < validation_x[0].size(); ++j) {
			*(x_val + i * validation_x[0].size() + j) = validation_x[i][j];
		}
	}

	vector<float> validation_y(training_labels.begin() + num_training, training_labels.end());
	for (int i = 0; i < validation_y.size(); ++i){
		*(y_val + i) = validation_y[i];
	}

	//subsampling for testing
	vector<vector<float> > test_x(test_images.begin(), test_images.begin() + num_test);
	for (int i = 0; i < test_x.size(); ++i){
		for (int j = 0; j < test_x[0].size(); ++j) {
			*(x_test + i * test_x[0].size() + j) = test_x[i][j];
		}
	}

	vector<float> test_y(test_labels.begin(), test_labels.begin() + num_test);
	for (int i = 0; i < test_y.size(); ++i){
		*(y_test + i) = test_y[i];
	}

	cout << "Number of training is: " << training_x.size() << endl;
	cout << "Number of validation is: " << validation_x.size() << endl;
	cout << "Number of testing is: " << test_x.size() << endl;
}

float calc_accuracy(vector<float> pred_labels, float *true_lables) {
	float yes = 0;

	for (int i = 0; i < pred_labels.size(); ++i) {
		if (pred_labels[i] == *(true_lables + i)) {
			yes += 1;
		}
	}
	
	return yes / pred_labels.size();
}

int main(int argc, char** argv) {
	int input_size = 32 * 32 * 3;
	int hidden_size = 50;
	int num_classes = 10;

	int num_training=49000;
	int num_validation=1000;
	int num_test=1000;

	DMatrix x, y, x_val, y_val, x_test, y_test;
	x.row = num_training;
	x.col = input_size;
	x.data = (float *)malloc(x.row * x.col * sizeof(float));
	memset(x.data, 0, x.row * x.col * sizeof(float));

	y.row = 1;
	y.col = num_training;
	y.data = (float *)malloc(y.row * y.col * sizeof(float));
	memset(y.data, 0, y.row * y.col * sizeof(float));

	x_val.row = num_validation;
	x_val.col = input_size;
	x_val.data = (float *)malloc(x_val.row * x_val.col * sizeof(float));
	memset(x_val.data, 0, x_val.row * x_val.col * sizeof(float));

	y_val.row = 1;
	y_val.col = num_validation;
	y_val.data = (float *)malloc(y_val.row * y_val.col * sizeof(float));
	memset(y_val.data, 0, y_val.row * y_val.col * sizeof(float));

	x_test.row = num_test;
	x_test.col = input_size;
	x_test.data = (float *)malloc(x_test.row * x_test.col * sizeof(float));
	memset(x_test.data, 0, x_test.row * x_test.col * sizeof(float));

	y_test.row = 1;
	y_test.col = num_test;
	y_test.data = (float *)malloc(y_test.row * y_test.col * sizeof(float));
	memset(y_test.data, 0, y_test.row * y_test.col * sizeof(float));

	load_cifar10(x.data, y.data, x_val.data, y_val.data, 
		         x_test.data, y_test.data, num_training, num_validation, num_test);


	TwoLayerNets *net = new TwoLayerNets(input_size, hidden_size, 
	                                 	 num_classes);
	clock_t clkStart;
    clock_t clkFinish;

	clkStart = clock();
	net->train(x, y, x_val, y_val, 1e-3, 0.95, 0.5, 1000);
	clkFinish = clock();

	cout << "Training time is : " << (float)(clkStart - clkFinish) / CLOCKS_PER_SEC << " secs" << '\n';

	vector<float> pred_lables = net->predict(x_val);

	float acc = calc_accuracy(pred_lables, y_val.data);
	cout << "The accuracy is: " << acc << endl;

	free(x.data);
	free(y.data);
	free(x_val.data);
	free(y_val.data);
	free(x_test.data);
	free(y_test.data);
	delete net;
}
