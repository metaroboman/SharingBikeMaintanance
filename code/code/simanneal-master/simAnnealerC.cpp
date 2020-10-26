#include <boost/python.hpp> 
using namespace boost::python; 
  
class Test 
{ 
public: 
 int Add(const int x, const int y) 
 { 
  return x + y; 
 } 
  
 int Del(const int x, const int y) 
 { 
  return x - y; 
 } 
}; 
  
BOOST_PYTHON_MODULE(test3) 
{ 
 class_<Test>("Test") 
  .def("Add", &Test::Add) 
  .def("Del", &Test::Del); 
}