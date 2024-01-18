#include "stdio.h"
#include "stdlib.h"
#include "time.h"

static const double e = 2.7182818;

double* generateWeights(int IO_count, int hidden_count, int layer_count){
    int total_weight_count = (IO_count * hidden_count) * (layer_count + 1);

    double* weights = malloc(total_weight_count * (sizeof(double)));

    for(int i = 0; i < total_weight_count; i++){
        weights[i] = (double)i / total_weight_count;
    }

    return weights;
}

double random(){
    return ((double) rand()) / RAND_MAX;
}

double* generateBias(int layer_count){
    int total_bias_count = layer_count + 1;

    double* biases = malloc(total_bias_count * (sizeof(double)));

    for(int i = 0; i < total_bias_count; i++){
        printf("%f\n", random());
    }

    return biases;
}

double* feedForward(
    int pre_nodes, 
    int post_nodes,
    double weights[],
    double biases[],
    double input[]
    ){

    double* result = malloc(post_nodes * sizeof(double));

    for(int i = 0; i < hidden_count; i++){
        for(int j = 0; j < IO_count; j++){
            result[i] = 
        }
    }
}

typedef struct{
    int IO_count;
    int hidden_count;
    int layer_count;
} Node;

typedef struct{
    double min;
    double max;
} Target;

int main(){
    srand(time(NULL));

    Node node;

    node.IO_count = 2;
    node.hidden_count = 2;
    node.layer_count = 1;

    double* input = malloc(node.IO_count * (sizeof(double)));

    input[0] = 0.05;
    input[1] = 0.10;

    double* weights = generateWeights(
        node.IO_count,
        node.hidden_count,
        node.layer_count
    );

    double* biases = generateBias(node.layer_count);

    Target target;

    target.max = 0.99;
    target.min = 0.01;

}

//https://www.javatpoint.com/pytorch-backpropagation-process-in-deep-neural-network