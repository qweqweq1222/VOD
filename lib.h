#pragma once
#include <iostream>
#include <cmath>
#include <map>
#include <fstream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/flann.hpp"
#include <experimental/filesystem>
#include <string>
#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;

const int MAX_FEATURES = 500;
const int MIN_FEATURES = 10;
const float GOOD_MATCH_PERCENT = 0.7f;
const float DEPTH_TRASH = 300.0f;
const int SAME_POINTS = 30;
const int NUM_OF_FRAMES = 500;
const int LOCAL_TRASH = 2;
const int START_KEY_FRAME = 0;

struct LeastSquareSolver {
	LeastSquareSolver(double a, double b, double c) : a(a), b(b), c(c) {};
	template <typename T>
	bool operator() (const T* const Pt2d, T* residuals) const {
		T pt[2];
		pt[0] = Pt2d[0];
		pt[1] = Pt2d[1];
		residuals[0] = pt[1] * T(a) + pt[0] * T(b) + T(c);
		return true;
	}
	static ceres::CostFunction* Create(double a,
		double b, double c) {
		return (new ceres::AutoDiffCostFunction< LeastSquareSolver, 1, 2>(
			new LeastSquareSolver(a, b, c)));
	}

	double a, b, c;
};
struct SnavelyReprojectionError {
	SnavelyReprojectionError(double observed_x, double observed_y, /*Mat Ti, Point3f pt,*/ double fx, double fy, double cx, double cy)
		: observed_x(observed_x), observed_y(observed_y), /*Ti(Ti), pt3d(pt),*/ fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const alpha_t,  
		const T* const point3d,
		T* residuals) const {

		T P3[3];
		T a = alpha_t[0];
		T b = alpha_t[1];
		T g = alpha_t[2];
		T dx = point3d[0] - alpha_t[3];
		T dy = point3d[1] - alpha_t[4];
		T dz = point3d[2] - alpha_t[5];
		P3[0] = T(cos(b) * cos(g)) * dx + T(sin(a) * sin(b)*cos(g) + sin(g) * cos(a)) * dy + T(sin(a)*sin(g) - sin(b)*cos(a)*cos(g)) * dz;
		P3[1] = T(-sin(g) * cos(b)) * dx + T(cos(a) * cos(g) - sin(a) * sin(b) * sin(g)) * dy + T(sin(a) * cos(g) + sin(b) * sin(g) * cos(a)) * dz;
		P3[2] = T(sin(b)) * dx + T(-sin(a) * cos(b)) * dy + T(cos(a) * cos(b)) * dz;
		T predicted_x = fx * P3[0] / P3[2] + cx;
		T predicted_y = fy * P3[1] / P3[2] + cy;
		T regulx = alpha_t[3];
		T reguly = alpha_t[4];
		T regulz = alpha_t[5];
		
		residuals[0] = (predicted_x - T(observed_x)); 
		residuals[1] = (predicted_y - T(observed_y));
		//residuals[2] = alpha_t[0];
		//residuals[3] = alpha_t[1];
		//residuals[4] = alpha_t[2];
		//residuals[5] = alpha_t[3];
		residuals[2] = alpha_t[4] * T(100);
		//residuals[7] = alpha_t[5];
		//residuals[8] = point3d[0];
		//residuals[9] = point3d[1];
		//residuals[10] = point3d[2];

		//std::cout << "observed_x: " << observed_x << " observed_y: " << observed_y << endl;
		//std::cout << "predicted_x: " << predicted_x << " predicted_y: " << predicted_y << endl;
		//std::cout << "-----------------------\n";
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y,/* Mat Ti, Point3f pt3d,*/ const double fx, const double fy, const double cx, const double cy) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 3, 6, 3>(
			new SnavelyReprojectionError(observed_x, observed_y, /*Ti, pt3d,*/ fx, fy, cx, cy)));
	}
	
	double observed_x;
	double observed_y;
	/*Mat Ti;
	Point3f pt3d;*/
	double fx, fy, cx, cy;
};
struct KeyPointMatches
{
	vector<DMatch> matches;
	vector<KeyPoint> kp1, kp2;

	KeyPointMatches(vector<DMatch> matches_, vector<KeyPoint> kp1_,
		vector<KeyPoint> kp2_) :matches(matches_), kp1(kp1_), kp2(kp2_) {};
	~KeyPointMatches() = default;
};
struct CameraInfo
{
	Mat cameraMatrix, rotMatrix, transVector;
	CameraInfo(Mat camera_matrix, Mat rot_matrix, Mat trans_vector) {
		camera_matrix.convertTo(camera_matrix, CV_32F, 1.0);
		rot_matrix.convertTo(rot_matrix, CV_32F, 1.0);
		trans_vector.convertTo(trans_vector, CV_32F, 1.0);

		cameraMatrix = camera_matrix;
		rotMatrix = rot_matrix;
		transVector = trans_vector;
	};
	~CameraInfo() = default;

};
struct ForOptimize
{
	Mat R, t;
	vector<Point3f> pts_3d;
	vector<Point2f> pts_2d;

	ForOptimize(Mat R_, Mat t_, vector<Point3f>& pts3d, vector<Point2f>& pts2d)
	{
		R_.convertTo(R_, CV_32F, 1.0);
		t_.convertTo(t_, CV_32F, 1.0);
		R = R_;
		t = t_;
		pts_3d = pts3d;
		pts_2d = pts2d;
	}
	~ForOptimize() = default;
};

KeyPointMatches AlignImages(Mat& im1, Mat& im2); // find features
CameraInfo Decompose(Mat proj_matrix); // decompose given P matrix of a camera

Mat CalculateDisparity(const cv::Mat& left_image, const cv::Mat& right_image);
Mat TAP(const Mat& R, const Mat& t); 
Mat T(Mat& R, Mat& t);

pair<Mat, Mat> EstimateMotion(Mat left, Mat right, Mat next, Mat P_left, Mat P_right);
pair<Mat, Mat> EstimateNoDynamicMotion(Mat left, Mat right, Mat next, Mat left_segment, Mat P_left, Mat P_right, std::vector<int> dynamic);
std::vector<std::vector<KeyPoint>> GetSamePoints(fs::directory_iterator left, fs::directory_iterator next, const int& N_features);
std::vector<double> Transform_vec(const Mat answer);

void VisualOdometry(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight);
void EstimateAndOptimize(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight);
void VisualNoDynamic(const std::string& left_path, const std::string& left_path_segment,  const std::string& right_path, const std::string& input, 
	const Mat& PLeft, const Mat& PRight, std::vector<int> dynamic);

/*bool operator != (KeyPoint kp1, KeyPoint kp2) {
	return !(kp1.pt.x == kp2.pt.x && kp1.pt.y == kp2.pt.y);
}
bool operator == (KeyPoint kp1, KeyPoint kp2) { return !(kp1 != kp2); }*/
