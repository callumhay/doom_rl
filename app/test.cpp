#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <tuple>
#include <memory>
#include <utility>
#include <numeric>
#include <unordered_map>

class MagicFoo {
public:
  int someInt;
  std::vector<int> vec;
  
  MagicFoo(int i) : someInt(i) {}
  MagicFoo(std::initializer_list<int> list) : MagicFoo(42) { // You can now call other constructors from the init list of another class constructor of the same class... so nice.
    for (std::initializer_list<int>::iterator it = list.begin(); it != list.end(); ++it) {
      vec.push_back(*it);
    }
  }
  //MagicFoo(int i, std::initializer_list<int> list) : MagicFoo(i), MagicFoo(list) {} // CAN'T DO THIS.


  virtual int doSomeMagic() { return this->someInt+1; }
  virtual void moreMagic() { this->someInt++; }
};

class SubclassMagicFoo : public MagicFoo {
  public:
    using MagicFoo::MagicFoo; // You can now inherit constructors from your parent class!

    virtual int doSomeMagic() override { return this->someInt+2; } // When overriding you should use the 'override' keyword!
    //virtual void notAOverride() override {...} // This will fail

    // We can finalize inheritance using the 'final' keyword - inheriting classes can no longer override this method
    virtual void moreMagic() final { this->someInt += 2; }
};

constexpr int fibonacci(const int n) {
  return n == 1 || n == 2 ? 1 : fibonacci(n-1) + fibonacci(n-2);
}

// It's possible to use auto as the return type :)
template<typename T, typename U> 
auto addWithAuto(T x, U y){
  return x + y;
}

const std::string myStr = "FOO!";
std::string lookup1() { return std::string("BLAH"); }
const std::string& lookupRef() { return myStr; }
// The importance of using decltype here is that you can return non-value (i.e., pointers, references, etc.)
// in a generic way based on compiler-time inference of the return type
decltype(auto) lookup1Wrap() { return lookup1(); }
decltype(auto) lookupRefWrap() { return lookupRef(); }

// constexpr if statements get simplified and have branch judgement made at compile time,
// e.g., the result of this example will be multiple functions with the respective types 
// that make calls to print_type_info without branches in them
template<typename T>
auto print_type_info(const T& t) {
  if constexpr (std::is_integral<T>::value) {
    return t + 1;
  } 
  else {
    return t + 0.001;
  }
}
// i.e., Results compile time functions:
// int print_type_info(const int& t) { return t + 1; }
// double print_type_info(const double& t) { return t + 0.001; }

// Forward Declarations (see below main() for implementation/definition)
void theBig5OfA();
void rvalueRefTest(); 
void valuePassing();

std::tuple<int, double, std::string> makeMeATuple() {
  return std::make_tuple(1, 2.3, "456");
}

template<typename T> class TD; // TD == "Type Displayer" - use this to show compiler errors for variable types
// e.g., auto x = <something>;
// TD<decltype(x)> xType; // Compiler error: "error: aggregate 'TD<**> xType' has incomplete type and cannot be defined", where ** is the deduced type

int main() {
  // Tuples
  {
    // There are three core functions for the use of tuples:
    // 1. std::make_tuple: construct tuple, and
    // 2. std::get: Get the value of a position in the tuple
    auto t = std::make_tuple(1, "Foo", 3.14);
    std::cout << std::get<0>(t) << ", " << std::get<1>(t) << ", " << std::get<2>(t) << std::endl;
    // NOTE: Since std::get uses templates, it expects compile-time indexing... which is annoying
    // e.g., int idx = 1; std::get<idx>(t); would be illegal since idx is initialized at runtime.

    // 3. std::tie: STL method of tuple unpacking
    int myInt;
    double myDbl;
    std::string myStr;
    std::tie(myInt, myDbl, myStr) = makeMeATuple();
    std::cout << myInt << ", " << myDbl << ", " << myStr << std::endl;

    // N.B., Use tuples mostly for returning multiple values from functions.
  }
  {
    // We can now do structured binding / unpacking of tuples and various containers...
    auto [a,b,c] = makeMeATuple();
    std::cout << a << ", " << b << ", " << c << std::endl;

    int myArr[3] = {1,2,3};
    auto [d,e,f] = myArr;  // copy unpacking
    d = 5;
    std::cout << myArr[0] << ", " << myArr[1] << ", " << myArr[2] << std::endl; // "1, 2, 3"

    auto& [dRef,eRef,fRef] = myArr; // reference unpacking
    dRef = 5;
    std::cout << myArr[0] << ", " << myArr[1] << ", " << myArr[2] << std::endl; // "5, 2, 3"
    
    auto&& [dRRef,eRRef,fRRef] = myArr; // reference unpacking (extra & doesn't make a difference here, still unpacked by reference)
    dRRef = 8;
    std::cout << myArr[0] << ", " << myArr[1] << ", " << myArr[2] << std::endl; // "8, 2, 3"

    std::array<int, 3> mySTLArr = {4,5,6};
    const auto& [x,y,z] = mySTLArr;
    std::cout << x << ", " << y << ", " << z << std::endl; // "4, 5, 6"

    // Structured binding only works if the structure is known at compile time. This is not the case for the vector.
    //std::vector<int> myVec = {1,2,3}; // fine.
    //auto [x,y,z] = myVec;             // totally illegal.
  }

  std::vector<int> vec = {1, 2, 3, 4};
  // since c++17, can be simplified by using `auto`
  const std::vector<int>::iterator itr = std::find(vec.begin(), vec.end(), 2);
  if (itr != vec.end()) {
    *itr = 3;
  }
  if (const std::vector<int>::iterator itr = std::find(vec.begin(), vec.end(), 3); itr != vec.end()) {
    *itr = 4;
  }

  // should output: 1, 4, 3, 4. can be simplified using `auto`
  for (std::vector<int>::iterator element = vec.begin(); element != vec.end(); ++element) {
    std::cout << *element << std::endl;
  }

  std::cout << fibonacci(10) << std::endl;

  MagicFoo foo({1,2,3}); // also works: "MagicFoo foo {1,2,3};"
  std::cout << "Initialization from another constructor: " << foo.someInt << std::endl;
  for (auto element = foo.vec.begin(); element != foo.vec.end(); ++element) {
    std::cout << *element << std::endl;
  }

  SubclassMagicFoo subclassFoo({4,5,6});
  std::cout << "Subclass Initialization from another constructor: " << subclassFoo.someInt << std::endl;
  for (auto element = subclassFoo.vec.cbegin(); element != subclassFoo.vec.cend(); ++element) {
    std::cout << *element << std::endl;
  }

  auto x = 1;
  auto y = 2;
  decltype(x+y) z; // Declare a type 'z' that is the same type as what the result of x+y would be

  // std::is_same is used to determine whether two types are the same/equal
  if (std::is_same<decltype(x), int>::value) {
    std::cout << "type x == int" << std::endl;
  }
  if (std::is_same<decltype(x), float>::value) {
    std::cout << "type x == float" << std::endl;
  }
  if (std::is_same<decltype(x), decltype(z)>::value) {
    std::cout << "type z == type x" << std::endl;
  }

  std::cout << "Adding with auto (10+23.3333): " << addWithAuto(10, 23.3333) << std::endl;
  std::cout << "Wrapping with decltype(auto) for regular string return: " << lookup1Wrap() << std::endl;
  std::cout << "Wrapping with decltype(auto) for reference string return: " << lookupRefWrap() << std::endl;


  std::cout << "constexpr with int type: " << print_type_info(10) << std::endl;
  std::cout << "constexpr with double type: " << print_type_info(2.3) << std::endl;

  // Range-based loop by reference!
  for (auto& element : vec) {
    element += 10;
  }
  // Range-based loops by value
  for (auto element : vec) {
    std::cout << element << std::endl;
  }


  // Lambda functions...
  {
    // Value capture is when the 'capture' variable is passed by value
    int value = 1;
    auto copy_value = [value] { return value; };
    auto compilerProvidedValueCapture = [=]{ return value; }; // We can let the compiler infer the current scope by value
    value = 100;
    auto stored_value = copy_value();
    // At this moment, stored_value == 1, and value == 100. i.e., copy_value copied 'value' when its was created.  
    std::cout << "stored_value = " << stored_value << std::endl;    
    std::cout << "more value capture: " << compilerProvidedValueCapture() << std::endl;
  }
  {
    // Reference capture is when the 'capture' variable is passed by reference
    int value = 1;
    auto copy_value = [&value] { return value; };
    auto compilerProvidedRefCapture = [&]{ return value; }; // We can let the compiler infer the current scope by reference
    value = 100;
    auto stored_value = copy_value();
    // At this moment, stored_value == 100, value == 100. i.e., copy_value stores a reference to 'value'.
    std::cout << "stored_value = " << stored_value << std::endl;
    std::cout << "more reference capture: " << compilerProvidedRefCapture() << std::endl;
  }
  {
    // Expression capture
    auto important = std::make_unique<int>(1);
    auto add = [v1 = 1, v2 = std::move(important)](int x, int y) -> int { return x+y+v1+(*v2); };
    std::cout << add(3,4) << std::endl; // Outputs '9' i.e., (3+4+1+1)
    //std::cout << *important << std::endl; // fails: EXC_BAD_ACCESS - important now stores a nullptr!
    // NOTE: if we call add again it would be bad since 'important' was already moved.
  }  
  {
    // Generic Lambda
    auto generic = [](auto x, auto y) { return x+y; };
    std::cout << generic(1, 2) << std::endl;
    std::cout << generic(1.1, 2.2) << std::endl;
  }

  // std::function allows us to wrap functions as l-values
  {
    int important = 10;
    std::function<int(int)> func = [&](int value) -> int {
      return 1+value+important;
    };
    std::cout << func(10) << std::endl; // "21"
  }
  {
    // HELPFUL: Think of lambdas as C structs with an overloaded () operator and with captures as its member variables
    // e.g., You cannot have a capture of [&&val] because that makes no sense as a member
    int lval = 1;
    auto lvalueLambda = [&lval]() { lval = 2; };
    lvalueLambda();
    std::cout << lval << std::endl; // "2"
  }

  {
    // lvalue and rvalue semantics...
    std::string lv1 = "string,";
    std::string&& rv1 = std::move(lv1); // convert an lvalue to an rvalue
    std::cout << rv1 << std::endl;      // "string,"
    const std::string& lv2 = lv1 + lv1; // legal, this will extend the lifetime of temporary variables
    std::cout << lv2 << std::endl;      // "string,string,"

    // Think of the T&& as a way of having const T& without the const... a way to modify without copying
    std::string&& rv2 = lv1 + lv2; // legal, rvalue ref extend lifecycle
    rv2 += "string";
    std::cout << rv2 << std::endl; // "string,string,string,string"
  }

  rvalueRefTest();

  {
    std::string str = "Hello world.";
    std::vector<std::string> v;

    v.push_back(str); // use push_back(const T&), causes a copy of str!
    std::cout << "str: " << str << std::endl; // "str: Hello world."

    v.push_back(std::move(str)); // use push_back(const T&&), no copy of str, the string will be moved to vector!
    std::cout << "str: " << str << std::endl;  // "" i.e., str is empty now because it was moved into v

    // print v...
    auto join = std::accumulate(v.begin() + 1, v.end(), v[0],
      [](const std::string& a, const std::string& b){ return a + " " + b; }
    );
    std::cout << join << std::endl; // Hello world. Hello world.
  }

  /*
  When in doubt about parameter passing of rvalue and lvalues, see this table:
  ==============================================================================
  Function parameter type | Argument parameter type | Post-derivation function parameter type
  T&                        lvalue ref                T&
  T&                        rvalue ref                T&
  T&&                       lvalue ref                T&
  T&&                       rvalue ref                T&&
  ==============================================================================
  */
  valuePassing();


  // std::unordered_(map|multimap|set|multiset) - hashtable implementations of older C++ containers!
  {
    std::unordered_map<int, std::string> umap = {
      {1, "1"}, {2, "2"}, {3, "3"}
    };
    // Iterates the same way as previous containers
    for (const auto& iter : umap) {
      std::cout << "Key:[" << iter.first << "] Value:[" << iter.second << "]" << std::endl;
    }
    // You can also iterate by [key,value]
    for (const auto& [key, value] : umap) {
      std::cout << "Key:[" << key << "] Value:[" << value << "]" << std::endl;
    }
    // ... or if you want to change stuff
    for (auto&& [key,value] : umap) {
      //key = 1; // illegal, key is a const in map iteration!
      value = "8";
    }
    for (const auto& [key, value] : umap) {
      std::cout << "Key:[" << key << "] Value:[" << value << "]" << std::endl; // all values are now "8"
    }
  }

  {
    // Shared Pointers (std::shared_ptr<T>)
    
  }


  // The big 5: See class A below
  theBig5OfA();
  
  
  int&& myInt = 5; // This is the same as saying "int myInt = 5;"
  std::cout << myInt << std::endl;
}

// rvalue References as move semantics (move constructor and move assignment operator)

// REMEMBER THE RULE OF "THE BIG FIVE":

// 1. Destructor [A::~A()] - Deallocate any memory that was initialized by A
// Always make sure every "new/new[]" call in the constructors has a matching "delete/delete[]"
// call in the destructor. If A is inherited from then the destructor must be virtual.

// 2. The assignment operator [A& A::operator=(const A& a)] - The assignment operator
// should perform a "deep" copy of the provided parameter 'a'. This way we don't just share
// memory addresses across instances (and cause issues with double deletes / hanging pointers).

// 3. The copy constructor [A::A(A& a)] - Similar to the assignment operator, we must perform
// a deep copy of the provided parameter 'a' to 'this'.

// 4. The move constructor [A::A(A&& a)] - When we don't want to duplicate the memory we can
// use the move constructor to MOVE all the memory from the provided 'a' into 'this'. The A&& signifies an rvalue reference.
// NOTE: The compiler only favours r-value reference overloads for modifiable r-values; for constant r-values
// it always prefers constant l-value references (This means overloading for const T&& has no real application)
// The move constructor:
// - Takes an r-value reference as a parameter.
// - Discards the object’s current state.
// - Transfers ownership of the r-value object into the receiver
// - Puts the r-value object into an ‘empty’ state.

// 5. The move assignment operator [A& A::operator=(A&& a)]
// Occasionally, it is useful to only have one resource at a time and to transfer ownership of that
// resource from one ‘manager’ to another (for example, this is what std::unique_ptr does). In such
// cases you may want to provide a move assignment operator.

class A {
public:
  int *pointer;

  A() : pointer(new int(1)) {
    std::cout << "default constructor [A()]: " << pointer << std::endl;
  }

  // NOTE: The 'explicit' keyword is used to avoid the following scenario from being valid:
  // void myFunc(A a) { ... }; myFunc(1);
  // 'explicit' will force the compilier to not automatically use the A(int) constructor.
  explicit A(int val) : pointer(new int(val)) {
    std::cout << "value constructor [A(int val)]: " << pointer << std::endl;
  }

  A(A& a) : pointer(new int(a.pointer != nullptr ? *a.pointer : 0)) {
    std::cout << "copy constructor [A(A& a)]: " << pointer << std::endl;
  }

  // IMPORTANT: If A inherits from some 'Base' then we need to std::move into the Base
  // e.g., A(A&& a) : Base(std::move(a)), pointer(a.pointer) { ... }
  A(A&& a) : pointer(a.pointer) { 
    a.pointer = nullptr;
    std::cout << "move constructor [A(A&& a)]: " << pointer << std::endl;
  }

  ~A() {
    std::cout << "destructor [~A()]: " << pointer << std::endl;
    delete pointer; // NOTE: Calling delete on a nullptr is totally fine - 'delete' checks for nullptr implicitly.
  }

  A& operator=(const A& a) {
    std::cout << "assignment operator [A& operator=(const A& a)]" << std::endl;
    *this->pointer = *a.pointer;
    return *this;
  }

  A& operator=(A&& a) {
    std::cout << "move assignment operator [A& operator=(A&& a)]" << std::endl;
    if (this != &a) { // ALWAYS CHECK FOR SELF-ASSIGNMENT!!!
      delete this->pointer;
      this->pointer = a.pointer;
      a.pointer = nullptr;
    }
    return *this;
  }
};

A returnRValueA() {
  return A();
}

void theBig5OfA() {
  A* a1 = new A(); // Default constructor
  A a2 = *a1;      // Assignment via copy constructor (NOT the assignment operator!)
  A a3(*a1);       // This also calls the copy constructor
  a3 = a2;         // Assignment operator

  A a4(std::move(a3));   // Move constructor - BECAREFUL, a3 is now emptied of its pointer (i.e., a3.pointer == nullptr)
  A a5(returnRValueA()); // This also calls the move constructor but the compiler will optimize out the call (implicit call)

  A a6;
  a6 = returnRValueA(); // Move assignment operator

  delete a1; // Destructor (also all other in-scope/stack A's will be destructed once the function exits)
}

A returnRValue(bool test) {
  // Awful... 2x default constructor calls, 2x destructor calls, 1x copy constructor
  A a,b;
  return test ? a : b; // equal to static_cast<A&&>(test ? a : b);
}

void rvalueRefTest() {
  A a1(1);
  A a2 = a1; // copy constructor
  A a3(a1);  // copy constructor
  A a4(std::move(a2)); // move constructor

  // IMPORTANT NOTE:
  // std::move doesn't actually do any moving; it just converts an l-value into an r-value. This forces the
  // compiler to use the object's move constructor (if defined) rather than the copy constructor.

  std::cout << a2.pointer << std::endl; // nullptr output "0x0" i.e., a2 is empty now!
  A&& aMove = std::move(a1); // move

  std::cout << "Calling returnRValue..." << std::endl;
  A obj = returnRValue(false); // No copy here! A(A&&) constructor is implicitly called :)
  std::cout << "obj: (ptr: " << obj.pointer << ", value: " << *obj.pointer << ")" << std::endl;
    
  std::cout << "exiting rvalueRefTest()" << std::endl;
  // When this exits all the destructors will be called for the various a* lvalues
}

// VERY IMPORTANT NOTE:
// The best heuristics for distinguishing an lvalue is that it is ADDRESSABLE.
// A key thing to remember is that ALL PARAMETERS ARE LVALUES even if the value fed
// into a function is an rvalue, the resulting parameter "T&& myParam" will be an lvalue.
void reference(int& v) {
  std::cout << "lvalue reference" << std::endl;
}
void reference(int&& v) {
  // At this moment, v, AS A PARAMETER, is an lvalue, however it is a reference to an rvalue
  std::cout << "rvalue reference" << std::endl;
}
template <typename T>
void pass(T&& v) {
  std::cout << "  normal param passing: ";
  reference(v);
  std::cout << "  std::move param passing: ";
  reference(std::move(v));
  std::cout << "  std::forward param passing: ";
  reference(std::forward<T>(v));
  std::cout << "  static_cast<T&&> param passing: ";
  reference(static_cast<T&&>(v)); // This is essentially the same thing as std::forward<T>!
}
void valuePassing() {
  std::cout << "rvalue pass:" << std::endl;
  pass(1);
  std::cout << "lvalue pass:" << std::endl;
  int l = 1;
  pass(l);
}
/*
// Here is what the implementation of std::forward looks like:
template<typename _Tp>
constexpr _Tp&& forward(typename std::remove_reference<_Tp>::type& __t) noexcept { return static_cast<_Tp&&>(__t); }

template<typename _Tp>
constexpr _Tp&& forward(typename std::remove_reference<_Tp>::type&& __t) noexcept {
  static_assert(!std::is_lvalue_reference<_Tp>::value, "template argument substituting _Tp is an lvalue reference type");
  return static_cast<_Tp&&>(__t);
}

===================================================================================================
VERY IMPORTANT!!!!!!
When std::forward accepts an lvalue, _Tp is deduced to the lvalue, so the return value is the lvalue;
and when it accepts the rvalue, _Tp is derived as an rvalue reference, and based on the collapse rule, the
return value becomes the rvalue of &&&&T, which collapses to &&T.
===================================================================================================
*/