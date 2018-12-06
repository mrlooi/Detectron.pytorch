#include "hough_voting.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hough_voting_forward", &hough_voting_forward, "hough_voting_forward");
  // m.def("backward", &hough_voting_backward, "hough_voting_backward");
}
