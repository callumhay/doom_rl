#ifndef __QNETWORKLSTM_HPP__
#define __QNETWORKLSTM_HPP__

#include <torch/torch.h>

class QNetworkLSTMImpl : public torch::nn::Module {
public:
  QNetworkLSTMImpl(size_t inputSize, size_t outputSize, size_t hiddenLSTMSize, size_t numLSTMLayers=1);

  torch::Tensor forward(torch::Tensor input);

private:
  torch::nn::LSTM lstm{nullptr};
  torch::nn::Sequential fcSeq{nullptr};

};

TORCH_MODULE(QNetworkLSTM);

#endif // __QNETWORKLSTM_HPP__