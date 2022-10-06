#include <iostream>
#include <cmath>
#include <fstream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/imgproc.hpp>
#include <experimental/filesystem>
#include <string>
#include<ceres/ceres.h>

namespace fs = std::experimental::filesystem;
using namespace cv;
const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.50f;


struct ForOptimize
{
	Mat R, t;
	std::vector<Point3d> pts_3d;
	std::vector<Point2d> pts_2d;

	ForOptimize(Mat& R, Mat& t, std::vector<Point3d>& pts3d, std::vector<Point2d>& pts2d) : R(R), t(t), pts_3d(pts3d), pts_2d(pts2d) {}
	~ForOptimize() = default;
};
struct Final
{
	std::vector<float*> observed;
	std::vector<float*> pt;
	float* R;
	float* t;

	Final(ForOptimize data) {
		for (int i = 0; i < data.pts_2d.size(); ++i) {
			float buffer[3] = { data.pts_3d[i].x,data.pts_3d[i].x, data.pts_3d[i].x };
			observed.push_back(buffer);
			float buffer_[2] = { data.pts_2d[i].x, data.pts_2d[i].y };
			pt.push_back(buffer_);
		}
		R = (float*)data.R.data;
		t = (float*)data.t.data;
	}
};
struct CameraInfo
{
	Mat cameraMatrix;
	Mat rotMatrix;
	Mat transVector;
	CameraInfo(Mat camera_matrix, Mat rot_matrix, Mat trans_vector) : cameraMatrix(camera_matrix),
		rotMatrix(rot_matrix), transVector(trans_vector) {};
	~CameraInfo() = default;

};
struct KeyPointMatches
{
	std::vector<DMatch> matches;
	std::vector<KeyPoint> kp1, kp2;

	KeyPointMatches(std::vector<DMatch> matches_, std::vector<KeyPoint> kp1_,
		std::vector<KeyPoint> kp2_) :matches(matches_), kp1(kp1_), kp2(kp2_) {};
	~KeyPointMatches() = default;
};

void Display(const Mat& img)
{
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
			std::cout << img.at<float>(i, j) << " ";
		std::cout << std::endl;
	}
}

KeyPointMatches alignImages(Mat& im1, Mat& im2) {
	Mat im1Gray, im2Gray, descriptors1, descriptors2;;
	std::vector<KeyPoint> keypoints1, keypoints2;
	std::vector<DMatch> matches;

	cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());

	std::sort(matches.begin(), matches.end());

	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());
	return KeyPointMatches(matches, keypoints1, keypoints2);
}
void FilterMatches(Mat& depth, std::vector<KeyPoint> kp_left, std::vector<DMatch>& matches, const float& trash_distance) {
	matches.erase(std::remove_if(matches.begin(), matches.end(),
		[&](const DMatch& match) {
			return depth.at<float>(int(kp_left.at(match.queryIdx).pt.y), int(kp_left.at(match.queryIdx).pt.x)) > trash_distance;
		}), matches.end());
}
Mat CalculateDisparity(const cv::Mat& left_image, const cv::Mat& right_image) {
	Mat l, r, disparity;
	int sad_window = 6;
	int num_disparities = sad_window * 16;
	int block_size = 11;

	cvtColor(left_image, l, cv::COLOR_BGR2GRAY);
	cvtColor(right_image, r, cv::COLOR_BGR2GRAY);

	cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, num_disparities, block_size,
		864, 3456, 0, 0, 0, 0, 0, 2);

	stereo->compute(l, r, disparity);
	disparity.convertTo(disparity, CV_64F, 1.0);
	disparity /= 16.0f;
	/*disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);*/
	return disparity;
}

CameraInfo Decompose(Mat proj_matrix)
{
	cv::Mat K(3, 3, cv::DataType<float>::type); // intrinsic parameter matrix
	cv::Mat R(3, 3, cv::DataType<float>::type); // rotation matrix
	cv::Mat T(4, 1, cv::DataType<float>::type);
	decomposeProjectionMatrix(proj_matrix, K, R, T);
	return { K, R, T };
}
Mat DepthMap(Mat& disparity, const Mat& K_left, const Mat& t_left, const Mat& t_right) {
	float f = K_left.at<float>(0, 0);
	float b = t_right.at<float>(0) - t_left.at<float>(0);

	for (int i = 0; i < disparity.rows; ++i)
		for (int j = 0; j < disparity.cols; ++j) {
			bool condition = disparity.at<float>(i, j) == 0.0 || disparity.at<float>(i, j) == -1.0;
			disparity.at<float>(i, j) = condition ? 0.1 : disparity.at<float>(i, j);
		}
	return f * b / disparity;
}
Mat StereoToDepth(const Mat& left, const Mat& right, const Mat& P0, const Mat& P1)
{
	Mat disparity = CalculateDisparity(left, right);
	CameraInfo cf1 = Decompose(P0);
	CameraInfo cf2 = Decompose(P1);

	return DepthMap(disparity, cf1.cameraMatrix, cf1.transVector, cf2.transVector);
}
ForOptimize EstimateMotion(Mat left, Mat right, Mat next, Mat P_left, Mat P_right) {

	std::vector<Point3d> object_points;
	std::vector<Point2d> image_points;
	srand(time(0));
	CameraInfo cil = Decompose(P_left);
	CameraInfo cir = Decompose(P_right);
	KeyPointMatches kpm = alignImages(left, next);
	Mat disparity = CalculateDisparity(left, right);
	Mat depth = StereoToDepth(left, right, P_left, P_right);
	FilterMatches(depth, kpm.kp1, kpm.matches, 100.0f);
	for (auto& match : kpm.matches) {
		Point2d pt_2d(kpm.kp2.at(match.trainIdx).pt.x, kpm.kp2.at(match.trainIdx).pt.y);
		float u = (kpm.kp1.at(match.queryIdx).pt.x);
		float v = (kpm.kp1.at(match.queryIdx).pt.y);
		float z = depth.at<float>(int(v), int(u));
		if (z < 60) {

			float x = z * (u - cir.cameraMatrix.at<float>(0, 2)) / cir.cameraMatrix.at<float>(0, 0);
			float y = z * (v - cir.cameraMatrix.at<float>(1, 2)) / cir.cameraMatrix.at<float>(1, 1);
			Point3d pt_3d(x, y, z);
			object_points.emplace_back(pt_3d);
			image_points.emplace_back(pt_2d);
		}
	}
	Mat R, t;
	cv::solvePnP(object_points, image_points, cil.cameraMatrix, cv::noArray(), R, t);
	return { R, t,object_points, image_points };
}
std::vector<Final> VisualOdometry(const std::vector<std::string>& left, const std::vector<std::string>& right,
	const std::vector<std::string>& next, const Mat& P_left, const Mat& P_right) {
	std::vector<Final> answer;
	for (int i = 0; i < left.size(); ++i) {
		ForOptimize buffer = EstimateMotion(imread(left[i]), imread(right[i]),
			imread(next[i]), P_left, P_right);
		answer.emplace_back(buffer);
		std::cout << "2d : " << buffer.pts_2d.size() << " 3d : " << buffer.pts_3d.size() << std::endl;
	}
	return answer;
}

void Test_Matches(Mat& lr, Mat& rr) {
	KeyPointMatches mt = alignImages(lr, rr);
	Mat result;
	drawMatches(lr, mt.kp1, rr, mt.kp2, mt.matches, result);
	imshow("w", result);
	waitKey(0);
}

void Test_Disparity(Mat& lr, Mat& rr) {
	Mat dsp = CalculateDisparity(lr, rr);
	Display(dsp);
	imshow("W", dsp);
	waitKey(0);
}
int main()
{
	std::string l = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_0/000000.png";
	std::string r = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_1/000000.png";
	std::string nl = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_0/000001.png";
	float P0[] = { 718.856f, 0.0f, 607.1928f, 0.0f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	float P1[] = { 718.856f, 0.0f, 607.1928f, -386.1448f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	Mat P_left(3, 4, cv::DataType<float>::type, P0);
	Mat P_right(3, 4, cv::DataType<float>::type, P1);
	/*

	cv::Mat img_l = cv::imread(l);
	cv::Mat img_r = cv::imread(r);
	cv::Mat next_img = cv::imread(nl);

	if (img_l.empty() || img_r.empty())
	{
		std::cout << "Could not read the image: " <<  std::endl;
		return 1;
	}


	float P0[] = { 718.856f, 0.0f, 607.1928f, 0.0f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	float P1[] = { 718.856f, 0.0f, 607.1928f, -386.1448f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	Mat P_left(3, 4, cv::DataType<float>::type, P0);
	Mat P_right(3, 4, cv::DataType<float>::type, P1);
	ForOptimize answer = EstimateMotion(img_l, img_r, next_img, P_left, P_right);
	Mat R;
	Rodrigues(answer.R, R, noArray());
	Display(R);
	std::cout << R.size() << std::endl;;
	std::cout << answer.t.size() << std::endl;*/

	std::string folder_left = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_0/";
	std::string folder_right = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_1";
	fs::directory_iterator left_iterator(folder_left);
	fs::directory_iterator right_iterator(folder_right);
	std::vector<ForOptimize> answers;
	std::vector<Mat> distance;
	ofstream in("C:/Users/Andrey/Desktop/Data/results.txt");
	Mat P(Size(4,4), CV_64FC1, Scalar(0));
	const int N = 10;
	
	for (int i = 0; i < N; ++i) {

		ForOptimize buffer = EstimateMotion(imread((*left_iterator).path().u8string()),
			imread((*right_iterator).path().u8string()),
			imread((*(++left_iterator)).path().u8string()), P_left, P_right);
		
		Mat S;
		Rodrigues(buffer.R, S, noArray());
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				P.at<float>(i, j) = S.at<float>(i, j);
		//Display(S);
		for (int i = 0; i < 3; ++i)
			P.at<float>(i, 3) = buffer.t.at<float>(0,i);
		P.at<float>(3, 3) = 1.0;
		//Mat N = P.inv();
		in << P.at<float>(0, 3) << " " << P.at<float>(1, 3) << " " << P.at<float>(2, 3) << std::endl;
		left_iterator++;
		right_iterator++;
	}

	return 0;
}
