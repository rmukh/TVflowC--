#include <iostream>
#include <chrono>
#include <math.h>
#include "TVflow/tvflow.hpp"

int main(int argc, char *argv[])
{
  if(argc < 5) {
    std::cout << "You have to specify all the required program arguments:" << std::endl;
    std::cout << "image type (0-grayscale, 1-rbg), image width (int), image height (int), input path(s) to csv(s)" << std::endl;
    return -1;
  }

  int W{std::stoi(argv[2])};
  int H{std::stoi(argv[3])};

  float T = 10.0;
  float dt = 1.0 / 50.0;

  int NOB = int(ceil(T / dt));

  int NIT = 100;
  float tol = 1e-5;

  float lami = 2 * dt;

  std::chrono::duration<double> diff{};
  if (std::stoi(argv[1]) == 0) {
    std::cout << "Grayscale image" << std::endl;
    
    Eigen::MatrixXd f = openData(argv[4]);

    auto start = std::chrono::steady_clock::now();
    Eigen::VectorXd S = run_TV_flow(f, W, H, NOB, lami, dt, tol, NIT);
    auto end = std::chrono::steady_clock::now();
    diff = end - start;

    saveData("S_test.csv", S);
  }
  else if (std::stoi(argv[1]) == 1) {
    std::cout << "RGB image" << std::endl;
    Eigen::MatrixXd r = openData(argv[4]);
    Eigen::MatrixXd g = openData(argv[5]);
    Eigen::MatrixXd b = openData(argv[6]);

    auto start = std::chrono::steady_clock::now();
    Eigen::MatrixX3d S = run_TV_flow_RGB(r, g, b, W, H, NOB, lami, dt, tol, NIT);
    auto end = std::chrono::steady_clock::now();
    diff = end - start;

    saveData("S_test.csv", S);
  }
  else {
    std::cout << "Wrong image type" << std::endl;
    return -1;
  }
  
  std::cout << "Time taken by the program: " << diff.count() << " seconds\n";
  return 0;
}
