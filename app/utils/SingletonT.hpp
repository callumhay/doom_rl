#ifndef __SINGLETONT_HPP__
#define __SINGLETONT_HPP__

#include <memory>

template<typename T>
class SingletonT {
public: 
  static std::unique_ptr<T>& getInstance() {
    if (instance == nullptr) {
      instance.reset(new T());
    }
    return instance;
  };

protected:
  SingletonT() = default;
  virtual ~SingletonT() = default;
  SingletonT(const SingletonT&) = delete;
  SingletonT& operator=(const SingletonT&) = delete;
  SingletonT(SingletonT&&) = delete;
  SingletonT& operator=(SingletonT&&) = delete;

private:
  inline static std::unique_ptr<T> instance = nullptr;
};

#endif // __SINGLETONT_HPP__
