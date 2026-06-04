#include "NeuralNetwork.hpp"
#include "Trace.hpp"
using namespace std;

void NeuralNetwork::eval() { evaluating = true; }
void NeuralNetwork::train() { evaluating = false; }
void NeuralNetwork::setLearningRate(double lr) { learningRate = lr; }
void NeuralNetwork::setInputNodeIds(vector<int> inputNodeIds) { this->inputNodeIds = inputNodeIds; }
void NeuralNetwork::setOutputNodeIds(vector<int> outputNodeIds) { this->outputNodeIds = outputNodeIds; }
vector<int> NeuralNetwork::getInputNodeIds() const { return inputNodeIds; }
vector<int> NeuralNetwork::getOutputNodeIds() const { return outputNodeIds; }

vector<double> NeuralNetwork::predict(DataInstance instance) {
    if (instance.x.size() != inputNodeIds.size()) return vector<double>();

    for (size_t i = 0; i < inputNodeIds.size(); ++i) 
        nodes.at(inputNodeIds.at(i))->postActivationValue = instance.x.at(i);

    // Layer-based BFT
    for (const auto& layer : layers) {
        for (int vId : layer) {
            visitPredictNode(vId);
            for (auto const& [destId, conn] : adjacencyList.at(vId)) {
                visitPredictNeighbor(conn);
            }
        }
    }
    vector<double> output;
    for (int id : outputNodeIds) output.push_back(nodes.at(id)->postActivationValue);

    if (evaluating) flush();
    else { batchSize++; contribute(instance.y, output.at(0)); }
    return output;
}

bool NeuralNetwork::contribute(double y, double p) {
    contributions.clear();
    for (int nodeId : outputNodeIds) {
        contribute(nodeId, y, p);
    }
    return true;
}

double NeuralNetwork::contribute(int nodeId, const double& y, const double& p) {
    visitContributeStart(nodeId); 
    if (contributions.count(nodeId)) return contributions.at(nodeId); 

    double outgoingContribution = 0;
    bool isOutput = false;
    for(int id : outputNodeIds) { if(id == nodeId) isOutput = true; }

    if (isOutput) {
        outgoingContribution = -1 * ((y - p) / (p * (1 - p)));
    } else {
        for (auto& [destId, conn] : adjacencyList.at(nodeId)) {
            double incoming = contribute(destId, y, p);
            visitContributeNeighbor(conn, incoming, outgoingContribution);
        }
    }
    visitContributeNode(nodeId, outgoingContribution);
    contributions[nodeId] = outgoingContribution;
    return outgoingContribution;
}

bool NeuralNetwork::update() {
    for (int i = 0; i < nodes.size(); i++) {
        nodes.at(i)->bias -= (learningRate * nodes.at(i)->delta);
        nodes.at(i)->delta = 0; 
    }
    for (int i = 0; i < adjacencyList.size(); i++) {
        for (auto& [destId, conn] : adjacencyList.at(i)) {
            conn.weight -= (learningRate * conn.delta);
            conn.delta = 0; 
        }
    }
    flush();
    return true;
}
