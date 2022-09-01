#ifndef __QNETWORKLSTM_HPP__
#define __QNETWORKLSTM_HPP__

#include <torch/torch.h>

class QNetworkLSTMImpl : public torch::nn::Module {
public:
  QNetworkLSTMImpl(size_t inputSize, size_t outputSize, size_t hiddenLSTMSize, size_t numLSTMLayers=2, double dropout=0.2) {

    this->lstm = torch::nn::LSTM(torch::nn::LSTMOptions(inputSize, hiddenLSTMSize)
      .num_layers(numLSTMLayers).batch_first(true).dropout(dropout)
    );
    this->fcOut = torch::nn::Linear(hiddenLSTMSize, outputSize);

    this->register_module("q_lstm0", this->lstm);
    this->register_module("fcOut", this->fcOut);
  }

  torch::Tensor forward(torch::Tensor input) {
    // NOTE: Inputs to the LSTM must satisfy the shape [i, (h0, c0)], where 
    // i: is the input tensor of shape (batchSize, sequenceLen, inputSize)
    // h0: is the initial hidden state tensor of shape (numLayers, batchSize, hiddenSize) - if not provided, defaults to all zeros
    // c0: is the initial cell state tensor of shape (numLayers, batchSize, hiddenSize) - if not provided, defaults to all zeros
    auto [lstmOutput, hncn] = this->lstm(input); // For now we just let the h0 and c0 values default to all zeros

    // NOTE: Output of the LSTM is a tensor of shape [o, (hn, cn)], i.e., [tensor, std::tuple(tensor,tensor)] where
    // o: is the output tensor of shape [batchSize, sequenceLen, hiddenSize]
    // hn: is a tensor of shape (numLayers, hiddenSize), containing the final hidden state
    // cn: is a tensor of shape (numLayers, hiddenSize), containing the final cell state

    // Get the last output in the sequence for all the batches
    using namespace torch::indexing;
    auto lastItemLstmOutput = lstmOutput.index({Slice(), -1, Slice()}); // i.e., [batchSize, hiddenSize] for the last element in the output sequence

    // Pass the last item LSTM output through the fully connected layer
    auto out = this->fcOut(lastItemLstmOutput);
    return out;
  }

private:
  torch::nn::LSTM lstm{nullptr};
  torch::nn::Linear fcOut{nullptr};

};

TORCH_MODULE(QNetworkLSTM);

#endif // __QNETWORKLSTM_HPP__