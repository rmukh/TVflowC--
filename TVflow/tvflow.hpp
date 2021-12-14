#ifndef tvflow_hpp
#define tvflow_hpp

#include <Eigen/Core>
#include <Eigen/Dense>

void grad(const Eigen::Ref<Eigen::MatrixXd> &img, Eigen::Ref<Eigen::ArrayXXd> out1, Eigen::Ref<Eigen::ArrayXXd> out2, int n, int m);
void div(const Eigen::Ref<Eigen::ArrayXXd> &g1, const Eigen::Ref<Eigen::ArrayXXd> &g2, Eigen::Ref<Eigen::ArrayXXd> out1, Eigen::Ref<Eigen::ArrayXXd> out2, int n, int m);
void tvdff(const Eigen::Ref<Eigen::MatrixXd> &f, Eigen::Ref<Eigen::MatrixXd> out, int n, int m, double lmd, double tol=1e-2, int iters=100);
void tvdff_color(const Eigen::Ref<Eigen::MatrixXd> &f_r, const Eigen::Ref<Eigen::MatrixXd> &f_g, const Eigen::Ref<Eigen::MatrixXd> &f_b, Eigen::Ref<Eigen::MatrixXd> out_r, Eigen::Ref<Eigen::MatrixXd> out_g, Eigen::Ref<Eigen::MatrixXd> out_b, int n, int m, double lmd, double tol=1e-2, int iters=100);
Eigen::VectorXd run_TV_flow(const Eigen::Ref<Eigen::MatrixXd> &f, int n, int m, int NOB, double lami, double dt, double tol=1e-2, int NIT=100);
Eigen::MatrixX3d run_TV_flow_RGB(const Eigen::Ref<Eigen::MatrixXd> &r, const Eigen::Ref<Eigen::MatrixXd> &g, const Eigen::Ref<Eigen::MatrixXd> &b, int n, int m, int NOB, double lami, double dt, double tol=1e-2, int NIT=100);
template <typename T>
extern void saveData(std::string fileName, const T &v);
Eigen::MatrixXd openData(std::string fileToOpen);

#endif /* tvflow_hpp */
