#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"

static const double e = 2.7182818;

typedef struct{
    int IO_count;
    int hidden_count;
    int layer_count;
} Node;

typedef struct{
    double min;
    double max;
} Target;

double* generateWeights(int IO_count, int hidden_count, int layer_count){
    int total_weight_count = 2 * (IO_count * hidden_count) + ((hidden_count * hidden_count) * (layer_count-1));

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
        biases[i] = random();
    }

    return biases;
}

double* generateTargets(int IO_count, Target *target){ 
    double* targets = malloc(IO_count * (sizeof(double)));
    targets[0] = target->min;

    double interval = (target->max - target->min) / (IO_count - 1);

    for(int i = 1; i <= IO_count - 2; i++){
        targets[i] = interval;
        interval += interval;
    }

    targets[IO_count-1] = target->max;

    return targets;
}

double sigmoid(double n){
    return 1 / (1 + (1 / powf(e, n)));
}

int* layerArray(Node *node, int total_layer_count){
    
    int* layers = malloc(total_layer_count * (sizeof(int)));

    layers[0] = node->IO_count;
    layers[total_layer_count - 1] = node->IO_count;

    for(int i = 1; i <= node->layer_count; i++){
        layers[i] = node->hidden_count;
    }

    return layers;
}

double* calculatePass(int pre_node_count, int post_node_count, double weights[], double bias, double inputs[], int start_weight){
    
    double* result = malloc(post_node_count * sizeof(double));

    int current_weight = start_weight;

    for(int i = 0; i < post_node_count; i++){
        double new_value = 0;

        for(int j = 0; j < pre_node_count; j++){
            new_value = new_value + (inputs[j] * weights[current_weight]);
            current_weight++;
        }
        result[i] = sigmoid(new_value + bias);
        printf("%f\n", result[i]);
    }

    printf("\r\n");

    return result;    
}

double* feedForward(Node *node, double weights[], double biases[], double inputs[]){

    int total_layer_count = 2 + node->layer_count;
    int* layers = layerArray(node, total_layer_count);
    int start_weight = 0;

    for(int i = 0; i < total_layer_count - 1; i++){
        double* result = calculatePass(layers[i], layers[i + 1], weights, biases[i], inputs, start_weight);

        start_weight += layers[i] * layers[i + 1];

        inputs = result;
    }

    return inputs;
}

double calculateTotalError(int IO_count, double values[], double targets[]){

    double error = 0;

    for(int i = 0; i < IO_count; i++){
        error += (double)(0.5 * (targets[i] - values[i])) * (double)(0.5 * (targets[i] - values[i]));
    }

    return error;
}

int main(){
    srand(time(NULL));

    Node node;

    node.IO_count = 3;
    node.hidden_count = 5;
    node.layer_count = 2;

    double* inputs = malloc(node.IO_count * (sizeof(double)));

    inputs[0] = 0.05;
    inputs[1] = 0.10;

    double* weights = generateWeights(
        node.IO_count,
        node.hidden_count,
        node.layer_count
    );

    double* biases = generateBias(node.layer_count);

    Target target;

    target.min = 0.01;
    target.max = 0.99;
    
    double* targets = generateTargets(node.IO_count, &target);

    double* final_values = feedForward(&node, weights, biases, inputs);

}

//https://www.javatpoint.com/pytorch-backpropagation-process-in-deep-neural-network