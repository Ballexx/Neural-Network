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

    double interval = 0;

    if(IO_count % 2 == 0){
        interval = (double)1 / (IO_count-1);
    }
    else{
        interval = (double)1 / (IO_count);
    }

    if(IO_count > 2){
        for(int i = 1; i < IO_count; i++){
            targets[i] = interval;
            interval += interval;
        }
    }

    targets[IO_count-1] = target->max;

    for(int i = 0; i < IO_count; i++){
        printf("%f\n", targets[i]);
    }

    return targets;
}

double sigmoid(double n){
    return 1 / (1 + (1 / powf(e, n)));
}

double* calculateHidden(int pre_node_count, int post_node_count, double weights[], double bias, double inputs[]){
    double* result = malloc(post_node_count * sizeof(double));

    int current_weight = 0;

    for(int i = 0; i < post_node_count; i++){
        double new_value = 0;

        for(int j = 0; j < pre_node_count; j++){
            new_value = new_value + (inputs[j] * weights[current_weight]);
            current_weight++;
        }
        result[i] = sigmoid(new_value + bias);
        printf("%f\n", result[i]);
    }

    return result;    
}

double* feedForward(Node *node, double weights[], double biases[], double inputs[]){
}

int main(){
    srand(time(NULL));

    Node node;

    node.IO_count = 4;
    node.hidden_count = 3;
    node.layer_count = 1;

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

    double* new_value = feedForward(&node, weights, biases, inputs);
}

//https://www.javatpoint.com/pytorch-backpropagation-process-in-deep-neural-network