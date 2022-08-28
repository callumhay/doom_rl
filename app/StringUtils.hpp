#ifndef __STRINGUTILS_HPP__
#define __STRINGUTILS_HPP__

#include <iostream>
#include <sstream>
#include <vector>

class StringUtils {
public:

  static std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delimiter)) { result.push_back(item); }
    return result;
  };

private:
  StringUtils(){};
};

#endif // __STRINGUTILS_HPP__
