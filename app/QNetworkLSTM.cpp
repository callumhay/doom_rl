
#include "QNetworkLSTM.hpp"

QNetworkLSTMImpl::QNetworkLSTMImpl(size_t inputSize, size_t outputSize, size_t hiddenLSTMSize, size_t numLSTMLayers=2) {

  this->lstm = torch::nn::LSTM(
    torch::nn::LSTMOptions(inputSize, hiddenLSTMSize)
      .num_layers(numLSTMLayers)
      .batch_first(true)
      .dropout(0.2)
  );

  /*
  this->fcSeq = torch::nn::Sequential({
    {'linear0',   torch::nn::Linear(hiddenLSTMSize, hiddenLSTMSize/2)},
    {'relu0',     torch::nn::ReLU()},
    {'linear1',   torch::nn::Linear(hiddenLSTMSize/2, hiddenLSTMSize/4)},
    {'relu1',     torch::nn::ReLU()},
    {'linearOut', torch::nn::Linear(hiddenLSTMSize/4, outputSize)}
  });
  */

  // Simplicity to start
  this->fcSeq = torch::nn::Sequential({
      {'linear0',   torch::nn::Linear(hiddenLSTMSize, outputSize)}
  });

  this->register_module('q_lstm0', this->lstm);
  this->register_module('fc_sequential', this->fcSeq);
}


torch::Tensor QNetworkLSTMImpl::forward(torch::Tensor input) {

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

  // Pass the last item LSTM output through the fully connected sequential network and return the output
  return this->fcSeq(lastItemLstmOutput);
}
