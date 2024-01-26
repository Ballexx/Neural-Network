#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"

static const double e = 2.7182818;
static const int debug_mode = 1;

typedef struct{
    int input_count;
    int output_count;
    int hidden_count;
    int layer_count;
} Node;

typedef struct{
    double min;
    double max;
} Target;

double* generateWeights(Node *node, int hidden_count, int layer_count){
    int total_weight_count = (node->input_count + node->output_count) * hidden_count + (hidden_count * hidden_count) * (layer_count-1);

    double* weights = malloc(total_weight_count * (sizeof(double)));

    for(int i = 0; i < total_weight_count; i++){
        weights[i] = (double)i / total_weight_count;
    }

    return weights;
}

double* generateBias(int layer_count){
    int total_bias_count = layer_count + 1;

    double* biases = malloc(total_bias_count * (sizeof(double)));

    for(int i = 0; i < total_bias_count; i++){
        biases[i] = (double)i / total_bias_count;
    }

    return biases;
}

double* generateTargets(int output_count, Target *target){ 
    double* targets = malloc(output_count * (sizeof(double)));
    targets[0] = target->min;

    double interval = (target->max - target->min) / (output_count - 1);

    for(int i = 1; i <= output_count - 2; i++){
        targets[i] = interval;
        interval += interval;
    }

    targets[output_count-1] = target->max;

    return targets;
}

double sigmoid(double n){
    return 1 / (1 + (1 / powf(e, n)));
}

int* layerArray(Node *node, int total_layer_count){
    
    int* layers = malloc(total_layer_count * (sizeof(int)));

    layers[0] = node->input_count;
    layers[total_layer_count - 1] = node->output_count;

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

        if(debug_mode == 1){
            printf("%0.10f\n", result[i]);
        }
    }

    if(debug_mode == 1){
        printf("\r\n");
    }

    return result;    
}

double* feedForward(Node *node, double weights[], double biases[], double inputs[]){

    int total_layer_count = 2 + node->layer_count;
    int* layers = layerArray(node, total_layer_count);
    int start_weight = 0;

    for(int i = 0; i < total_layer_count - 1; i++){
        double* result = calculatePass(
            layers[i], 
            layers[i + 1], 
            weights, 
            biases[i], 
            inputs, 
            start_weight
        );

        start_weight += layers[i] * layers[i + 1];

        inputs = result;
    }

    return inputs;
}

double calculateTotalError(int output_count, double values[], double targets[]){

    double error = 0;

    for(int i = 0; i < output_count; i++){
        error += (double)0.5 * pow(targets[i] - values[i], 2);
    }

    if(debug_mode == 1){
        printf("\nTotal error: %0.10f\r\n", error);
    }

    return error;
}

double* backPropagate(Node *node, double new_values[], double error, double targets[]){

    for(int i = 0; i < node->output_count; i++){
        
        double x = (double)-(targets[i] - new_values[i]);

        double y = new_values[i] * (1 - new_values[i]);

        double z = 

    }

}

int main(){
    srand(time(NULL));

    Node node;

    node.input_count = 2;
    node.output_count = 2;
    node.hidden_count = 2;
    node.layer_count = 1;

    double* inputs = malloc(node.input_count * (sizeof(double)));

    inputs[0] = 0.05;
    inputs[1] = 0.10;
    inputs[2] = 0.15;

    double* weights = generateWeights(
        &node,
        node.hidden_count,
        node.layer_count
    );

    double* biases = generateBias(node.layer_count);

    Target target;

    target.min = 0.01;
    target.max = 0.99;
    
    double* targets = generateTargets(node.output_count, &target);

    double* final_values = feedForward(&node, weights, biases, inputs);
    double error = calculateTotalError(node.output_count, final_values, targets);
}

//https://www.javatpoint.com/pytorch-backpropagation-process-in-deep-neural-network