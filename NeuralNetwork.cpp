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

    for (size_t i = 0; i < inputNodeIds.size(); ++i) {
        nodes.at(inputNodeIds.at(i))->postActivationValue = instance.x.at(i);
    }

    for (const auto& layer : layers) {
        for (int vId : layer) {
            bool isInput = false;
            for(int id : inputNodeIds) if(id == vId) isInput = true;

            if (!isInput) {
                visitPredictNode(vId); 
            }
            for (auto const& [destId, conn] : adjacencyList.at(vId)) {
                visitPredictNeighbor(conn); 
            }
        }
    }

    vector<double> output;
    for (int id : outputNodeIds) output.push_back(nodes.at(id)->postActivationValue);


    if (evaluating) {
        flush(); 
    } else { 
        batchSize++; 
        contribute(instance.y, output.at(0)); 
    }
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

NeuralNetwork::NeuralNetwork() : Graph(0) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(int size) : Graph(size) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(string filename) : Graph() {
    // open file
    ifstream fin(filename);

    // error check
    if (fin.fail()) {
        cerr << "Could not open " << filename << " for reading. " << endl;
        exit(1);
    }

    // load network
    loadNetwork(fin);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;

    // close file
    fin.close();
}

NeuralNetwork::NeuralNetwork(istream& in) : Graph() {
    loadNetwork(in);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

const vector<vector<int> >& NeuralNetwork::getLayers() const {
    return layers;
}

void NeuralNetwork::loadNetwork(istream& in) {
    int numLayers(0), totalNodes(0), numNodes(0), weightModifications(0), biasModifications(0); string activationMethod = "identity";
    string junk;
    in >> numLayers; in >> totalNodes; getline(in, junk);
    if (numLayers <= 1) {
        cerr << "Neural Network must have at least 2 layers, but got " << numLayers << " layers" << endl;
        exit(1);
    }

    // resize network to accomodate expected nodes.
    resize(totalNodes);
    this->size = totalNodes;

    int currentNodeId(0);

    vector<int> previousLayer;
    vector<int> currentLayer;
    for (int i = 0; i < numLayers; i++) {
        currentLayer.clear();
        //  For each layer

        // get nodes for this layer and activation method
        in >> numNodes; in >> activationMethod; getline(in, junk);

        for (int j = 0; j < numNodes; j++) {
            // For every node, add a new node to the network with proper activationMethod
            // initialize bias to 0.
            updateNode(currentNodeId, NodeInfo(activationMethod, 0, 0));
            // This node has an id of currentNodeId
            currentLayer.push_back(currentNodeId++);
        }

        if (i != 0) {
            // There exists a previous layer, now we set out connections
            for (int k = 0; k < previousLayer.size(); k++) {
                for (int w = 0; w < currentLayer.size(); w++) {

                    // Initialize an initial weight of a sample from the standard normal distribution
                    updateConnection(previousLayer.at(k), currentLayer.at(w), sample());
                }
            }
        }

        // Crawl forward.
        previousLayer = currentLayer;
        layers.push_back(currentLayer);
    }
    in >> weightModifications; getline(in, junk);
    int v(0),u(0); double w(0), b(0);

    // load weights by updating connections
    for (int i = 0; i < weightModifications; i++) {
        in >> v; in >> u; in >> w; getline(in , junk);
        updateConnection(v, u, w);
    }

    in >> biasModifications; getline(in , junk);

    // load biases by updating node info
    for (int i = 0; i < biasModifications; i++) {
        in >> v; in >> b; getline(in, junk);
        NodeInfo* thisNode = getNode(v);
        thisNode->bias = b;
    }

    setInputNodeIds(layers.at(0));
    setOutputNodeIds(layers.at(layers.size()-1));
}

// visitPredictNode: called when your BFT dequeues a node.
// It completes the computation for this node by:
//   1. Adding the bias to the accumulated weighted sum (preActivationValue)
//   2. Applying the activation function and storing the result in postActivationValue
// After this call, the node's output value (postActivationValue) is ready to be
// passed forward to the next layer via visitPredictNeighbor.
void NeuralNetwork::visitPredictNode(int vId) {
    // accumulate bias, and activate
    NodeInfo* v = nodes.at(vId);
    v->preActivationValue += v->bias;
    v->activate();
    // visualization use
    if (viz::isTracing()) {
        viz::traceNodeState(0, "forward", vId,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "current");
    }
}

// visitPredictNeighbor: called for each outgoing connection from a dequeued node.
// It accumulates one term of the weighted sum into the destination node:
//   dest.preActivationValue += source.postActivationValue * weight
// This must be called for ALL incoming connections to a node before
// visitPredictNode is called on that node — which is why BFT is required:
// it ensures a whole layer's outputs are ready before the next layer is processed.
void NeuralNetwork::visitPredictNeighbor(Connection c) {
    NodeInfo* v = nodes.at(c.source);
    NodeInfo* u = nodes.at(c.dest);
    double w = c.weight;
    u->preActivationValue += v->postActivationValue * w;
    // visualization use
    if (viz::isTracing()) {
        viz::traceEdgeState(0, "forward",
                            c.source,
                            c.dest,
                            c.weight,
                            c.delta);
        viz::traceNodeState(0, "forward", c.dest,
                            u->preActivationValue,
                            u->postActivationValue,
                            u->bias,
                            u->delta,
                            "neighbor");
    }
}

// visitContributeStart: called at the start of the contribution step for a node.
void NeuralNetwork::visitContributeStart(int vId) {
    NodeInfo* v = nodes.at(vId);
    // visualization use
    if (viz::isTracing()) {
        viz::traceNodeState(0, "backward", vId,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "stack");
    }
}
// visitContributeNode: called after all neighbors of a node have been visited during DFT.
// outgoingContribution at this point holds the sum of weighted incoming contributions
// from the next layer. This function:
//   1. Multiplies outgoingContribution by the activation derivative at this node
//      (chain rule: how much did this node's activation affect the error?)
//   2. Accumulates that result into the node's delta (gradient for its bias)
// After this call, outgoingContribution holds the value to be passed back to
// the previous layer as their incomingContribution.
void NeuralNetwork::visitContributeNode(int vId, double& outgoingContribution) {
    NodeInfo* v = nodes.at(vId);
    outgoingContribution *= v->derive();


    v->delta += outgoingContribution;
}

// visitContributeNeighbor: called for each outgoing connection during DFT, before visitContributeNode.
// incomingContribution is the contribution returned by the recursive call on the neighbor (next layer).
// This function:
//   1. Adds weight * incomingContribution to outgoingContribution
//      (this node's share of the error flowing back from the neighbor)
//   2. Accumulates the weight gradient into c.delta
//      (how much should this weight change? proportional to incomingContribution * this node's output)
void NeuralNetwork::visitContributeNeighbor(Connection& c, double& incomingContribution, double& outgoingContribution) {
    NodeInfo* v = nodes.at(c.source);
    
    if (v->postActivationValue == 0) {
        std::cerr << "CRITICAL: Node " << c.source << " has 0 postActivationValue!" << std::endl;
    }
    outgoingContribution += c.weight * incomingContribution;

    c.delta += incomingContribution * v->postActivationValue;
}

void NeuralNetwork::flush() {
    // set every node value to 0 to refresh computation.
    for (int i = 0; i < nodes.size(); i++) {
        nodes.at(i)->postActivationValue = 0;
        nodes.at(i)->preActivationValue = 0;
    }
    contributions.clear();
    batchSize = 0;
}

double NeuralNetwork::assess(string filename) {
    DataLoader dl(filename);
    return assess(dl);
}

double NeuralNetwork::assess(DataLoader dl) {
    bool stateBefore = evaluating;
    evaluating = true;
    double count(0);
    double correct(0);
    vector<double> output;
    for (int i = 0; i < dl.getData().size(); i++) {
        DataInstance di = dl.getData().at(i);
        output = predict(di);
        if (static_cast<int>(round(output.at(0))) == di.y) {
            correct++;
        }
        count++;
    }

    if (dl.getData().empty()) {
        cerr << "Cannot assess accuracy on an empty dataset" << endl;
        exit(1);
    }
    evaluating = stateBefore;
    return correct / count;
}


void NeuralNetwork::saveModel(string filename) {
    ofstream fout(filename);
    
    fout << layers.size() << " " << getNodes().size() << endl;
    for (int i = 0; i < layers.size(); i++) {
        NodeInfo* layerNode = getNodes().at(layers.at(i).at(0));
        string activationType = getActivationIdentifier(layerNode->activationFunction);

        fout << layers.at(i).size() << " " << activationType << endl;
    }

    int numWeights = 0;
    int numBias = 0;
    stringstream weightStream;
    stringstream biasStream;
    for (int i = 0; i < nodes.size(); i++) {
        numBias++;
        biasStream << i << " " << nodes.at(i)->bias << endl;

        for (auto j = adjacencyList.at(i).begin(); j != adjacencyList.at(i).end(); j++) {
            numWeights++;
            weightStream << j->second.source << " " << j->second.dest << " " << j->second.weight << endl;
        }
    }

    fout << numWeights << endl;
    fout << weightStream.str();
    fout << numBias << endl;
    fout << biasStream.str();

    fout.close();


}

ostream& operator<<(ostream& out, const NeuralNetwork& nn) {
    for (int i = 0; i < nn.layers.size(); i++) {
        out << "layer " << i << ": ";
        for (int j = 0; j < nn.layers.at(i).size(); j++) {
            out << nn.layers.at(i).at(j) << " ";
        }
        out << endl;
    }
    // outputs the nn in dot format
    out << static_cast<const Graph&>(nn) << endl;
    return out;
}
