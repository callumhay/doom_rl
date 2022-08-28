#ifndef __CONVUTILS_HPP__
#define __CONVUTILS_HPP__

class ConvUtils {
public:
  static int calcShapeConv2dOneDim(int inputSize, int kernelSize, int stride, int padding=0) {
    return (inputSize-kernelSize + 2*padding) / stride + 1;
  };

  using Conv2dOutputShape = std::array<int, 3>; // {width, height, channels}
  static auto calcShapeConv2d(int inputWidth, int inputHeight, int outputChannels, int kernelSize, int stride, int padding=0) {
    return Conv2dOutputShape({
      ConvUtils::calcShapeConv2dOneDim(inputWidth,  kernelSize, stride, padding),
      ConvUtils::calcShapeConv2dOneDim(inputHeight, kernelSize, stride, padding),
      outputChannels
    });
  };

private:
  ConvUtils(){};
};

#endif // __CONVUTILS_HPP__