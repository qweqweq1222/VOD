#include <iostream>
#include <cmath>
#include <fstream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/imgproc.hpp>
#include <experimental/filesystem>
#include <string>
#include <vector>
#include<ceres/ceres.h>

namespace fs = std::experimental::filesystem;
using namespace cv;
const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.50f;
const float DEPTH_TRASH = 200.0f;



bool operator == (KeyPoint l, KeyPoint r) { return int(l.pt.x) == int(r.pt.x) && int(l.pt.y) == int(r.pt.y); }
bool operator != (KeyPoint l, KeyPoint r) { return !(l == r); }


struct SnavelyReprojectionError {
	SnavelyReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy)
		: observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const Rt,
		const T* const point3d,
		T* residuals) const {

		T R[9];
		T t[3];
		T P[3];
		for (int i = 0; i < 9; ++i)
			R[i] = Rt[i];
		for (int i = 9; i < 12; ++i)
			t[i-9] = Rt[i];

		P[0] = R[0] * point3d[0] + R[1] * point3d[1] + R[2] * point3d[2] + t[0];
		P[1] = R[3] * point3d[0] + R[4] * point3d[1] + R[5] * point3d[2] + t[1];
		P[2] = R[6] * point3d[0] + R[7] * point3d[1] + R[8] * point3d[2] + t[2];

		P[0] /= P[2];
		P[1] /= P[2];
		T predicted_x = fx * P[0] + cx;
		T predicted_y = fy * P[1] * cy;

		// The error is the difference between the predicted and observed position.
		residuals[0] = predicted_x - T(observed_x);
		residuals[1] = predicted_y - T(observed_y);
		return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double fx, const double fy, const double cx, const double cy) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 12, 3>(
			new SnavelyReprojectionError(observed_x, observed_y, fx, fy, cx, cy)));
	}

	double observed_x;
	double observed_y;
	double fx, fy, cx, cy;
};

struct SnavelyReprojectionErrorFixedPoints {
	SnavelyReprojectionErrorFixedPoints(double observed_x, double observed_y, 
										double pts3d_x, double pts3d_y, double pts3d_z, 
										double fx, double fy, double cx, double cy)
		: observed_x(observed_x), observed_y(observed_y), pts3d_x(pts3d_x),
		  pts3d_y(pts3d_y), pts3d_z(pts3d_z), fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const Rt,
		T* residuals) const {

		T R[9];
		T t[3];
		T P[3];
		for (int i = 0; i < 9; ++i)
			R[i] = Rt[i];
		for (int i = 9; i < 12; ++i)
			t[i - 9] = Rt[i];

		P[0] = R[0] * pts3d_x + R[1] * pts3d_y + R[2] * pts3d_z + t[0];
		P[1] = R[3] * pts3d_x + R[4] * pts3d_y + R[5] * pts3d_z + t[1];
		P[2] = R[6] * pts3d_x + R[7] * pts3d_y + R[8] * pts3d_z + t[2];

		P[0] /= P[2];
		P[1] /= P[2];
		T predicted_x = fx * P[0] + cx;
		T predicted_y = fy * P[1] * cy;

		// The error is the difference between the predicted and observed position.
		residuals[0] = predicted_x - T(observed_x);
		residuals[1] = predicted_y - T(observed_y);
		return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(double observed_x, double observed_y,
		double pts3d_x, double pts3d_y, double pts3d_z,
		double fx, double fy, double cx, double cy) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorFixedPoints, 2, 12>(
			new SnavelyReprojectionErrorFixedPoints(observed_x, observed_y, pts3d_x, pts3d_y, 
				pts3d_z, fx, fy, cx, cy)));
	}

	double observed_x, observed_y;
	double pts3d_x, pts3d_y, pts3d_z;
	double fx, fy, cx, cy;
};
struct ForOptimize
{
	Mat R, t;
	std::vector<Point3f> pts_3d;
	std::vector<Point2f> pts_2d;

	ForOptimize(Mat R_, Mat t_, std::vector<Point3f>& pts3d, std::vector<Point2f>& pts2d)
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

KeyPointMatches AlignImages(Mat& im1, Mat& im2) {
	Mat im1Gray, im2Gray, descriptors1, descriptors2;;
	std::vector<KeyPoint> keypoints1, keypoints2;
	std::vector<std::vector<DMatch>> matches;

	cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
	matcher->knnMatch(descriptors1, descriptors2, matches, 2);

	
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < matches.size(); i++)
		if (matches[i][0].distance < GOOD_MATCH_PERCENT * matches[i][1].distance)
			good_matches.push_back(matches[i][0]);
	std::sort(good_matches.begin(), good_matches.end());
	const int numGoodMatches = good_matches.size() * GOOD_MATCH_PERCENT;
	good_matches.erase(good_matches.begin() + numGoodMatches, good_matches.end());

	/*Mat img_matches;
	drawMatches(im1, keypoints1, im2, keypoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	--Show detected matches
	Size size(img_matches.size().width/1.3, img_matches.size().height /1.3);
	Mat im;
	resize(img_matches,im,size);
	imshow("Good Matches", im);
	waitKey();*/
	return KeyPointMatches(good_matches, keypoints1, keypoints2);
}
void FilterMatches(Mat& depth, std::vector<KeyPoint> kp_left, std::vector<DMatch>& matches, const float& trash_distance) {
	matches.erase(std::remove_if(matches.begin(), matches.end(),
		[&](const DMatch& match) {
			return depth.at<float>(int(kp_left.at(match.queryIdx).pt.y), int(kp_left.at(match.queryIdx).pt.x)) > DEPTH_TRASH;
		}), matches.end());
}
Mat CalculateDisparity(const cv::Mat& left_image, const cv::Mat& right_image) {
	Mat l, r, disparity;
	int sad_window = 6;
	int num_disparities = sad_window * 16;
	int block_size = 11;

	cvtColor(left_image, l, cv::COLOR_BGR2GRAY);
	cvtColor(right_image, r, cv::COLOR_BGR2GRAY);

	cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, num_disparities, block_size, 864, 3456, 0, 0, 0, 0, 0, 2);

	stereo->compute(l, r, disparity);
	disparity.convertTo(disparity, CV_32F, 1.0);
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

	float coeff = K_left.at<float>(0, 0)*(t_right.at<float>(0) - t_left.at<float>(0));

	for (int i = 0; i < disparity.rows; ++i)
		for (int j = 0; j < disparity.cols; ++j) {
			bool condition = disparity.at<float>(i, j) == 0.0 || disparity.at<float>(i, j) == -1.0;
			disparity.at<float>(i, j) = condition ? 0.1 : disparity.at<float>(i, j);
		}

	return coeff / disparity;
}
Mat StereoToDepth(const Mat& left, const Mat& right, const Mat& P0, const Mat& P1)
{
	Mat disparity = CalculateDisparity(left, right);
	CameraInfo cf1 = Decompose(P0);
	CameraInfo cf2 = Decompose(P1);

	return DepthMap(disparity, cf1.cameraMatrix, cf1.transVector, cf2.transVector);
}
ForOptimize EstimateMotion(Mat left, Mat right, Mat next, Mat P_left, Mat P_right) {

	std::vector<Point3f> object_points;
	std::vector<Point2f> image_points;
	CameraInfo cil = Decompose(P_left);
	CameraInfo cir = Decompose(P_right);
	KeyPointMatches kpm = AlignImages(left, next);
	Mat depth = StereoToDepth(left, right, P_left, P_right);
	FilterMatches(depth, kpm.kp1, kpm.matches, DEPTH_TRASH);
	std::sort(kpm.matches.begin(), kpm.matches.end());
	//std::vector<DMatch> result(3);
	//copy(kpm.matches.begin(), kpm.matches.begin() + 3, result.begin());
	for (auto& match : kpm.matches) {

		Point2f pt_2d(kpm.kp2.at(match.trainIdx).pt.x, kpm.kp2.at(match.trainIdx).pt.y);
		float u = (float(kpm.kp1.at(match.queryIdx).pt.x)); /* x -> cols, y-> rows => u->cols, v->rows */
		float v = (float(kpm.kp1.at(match.queryIdx).pt.y));
		float z = depth.at<float>(int(v), int(u));

		float x = z * (u - cil.cameraMatrix.at<float>(0, 2)) / cil.cameraMatrix.at<float>(0, 0);
		float y = z * (v - cil.cameraMatrix.at<float>(1, 2)) / cil.cameraMatrix.at<float>(1, 1);
		object_points.emplace_back(Point3f{ x, y, z });
		image_points.emplace_back(pt_2d);
	
	}
	Mat R, t;
	solvePnPRansac(object_points, image_points, cil.cameraMatrix, noArray(), R, t);
	return { R, t, object_points, image_points };
}

std::vector<std::vector<KeyPoint>> NoName(fs::directory_iterator left, fs::directory_iterator next, const int& N_features)
{

	Mat l = imread((*left).path().u8string());
	Mat n = imread((*next).path().u8string());

	std::vector<std::vector<KeyPoint>> final_res;

	KeyPointMatches ln = AlignImages(l, n);
	std::vector<KeyPoint> lt, nt;

	for (auto& m : ln.matches) {
		lt.push_back(ln.kp1.at(m.queryIdx));
		nt.push_back(ln.kp2.at(m.trainIdx));
	}

	final_res.push_back(lt);
	final_res.push_back(nt);

	while (final_res[0].size() >= N_features) 
	{
		++left;
		++next;

		l = imread((*left).path().u8string());
		n = imread((*next).path().u8string());

		KeyPointMatches buffer = AlignImages(l, n);
		std::vector<KeyPoint> adder;
		std::vector<KeyPoint> copy = final_res[final_res.size() - 1];
		for (auto & p : copy) {

			auto it = std::find_if(begin(buffer.matches), end(buffer.matches), 
				[&](DMatch m) { return buffer.kp1.at(m.queryIdx) == p; });

			auto it_ = std::find_if(final_res[final_res.size() - 1].begin(), final_res[final_res.size() - 1].end(), 
									[&](KeyPoint P) { return P == p; });

			int idx = it_ - final_res[final_res.size() - 1].begin();

			if (it != buffer.matches.end())
				adder.push_back(buffer.kp2.at((*it).trainIdx));
			else
				for (auto& el : final_res)
					el.erase(el.begin() + idx);

			if (final_res[0].size() <= N_features)
				break;
		}
		
		if (final_res[0].size() <= N_features)
			break;
		else
			final_res.push_back(adder);
	}
	return final_res;
}


Mat T(Mat& R, Mat& t) {

	Mat T(Size(4, 4), CV_32F, Scalar(0));

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			T.at<float>(i, j) = R.at<float>(i, j);

	for (int i = 0; i < 3; ++i)
		T.at<float>(i, 3) = t.at<float>(i);

	T.at<float>(3, 3) = 1.0;
	
	return T;
}

std::vector<double*> CalculateDepth(const Mat& l, const Mat& r, const std::vector<std::vector<KeyPoint>>&kp, const Mat& Pl, const Mat& Pr) {

	std::vector<double*> pts3d;
	CameraInfo cf1 = Decompose(Pl);
	CameraInfo cf2 = Decompose(Pr);
	float coeff = cf1.cameraMatrix.at<float>(0, 0) * (cf2.transVector.at<float>(0) - cf1.transVector.at<float>(0));
	Mat dsp = CalculateDisparity(l, r);
	dsp.convertTo(dsp, CV_32F);
	for (auto& p : kp[0]) {
		double *mas = new double [3];
		float u = p.pt.x; /* x -> cols, y-> rows => u->cols, v->rows */
		float v = p.pt.y;
		float z = coeff / dsp.at<float>(int(v),int(u));

		float x = z * (u - cf1.cameraMatrix.at<float>(0, 2)) / cf1.cameraMatrix.at<float>(0, 0);
		float y = z * (v - cf1.cameraMatrix.at<float>(1, 2)) / cf1.cameraMatrix.at<float>(1, 1);
		mas[0] = x;
		mas[1] = y;
		mas[2] = z;
		pts3d.push_back(mas);
	}
	return pts3d;

}

double* Transform(const Mat answer)
{
	double *Rt = new double[12];
	Rt[0] = answer.at<float>(0, 0);
	Rt[1] = answer.at<float>(0, 1);
	Rt[2] = answer.at<float>(0, 2);
	Rt[3] = answer.at<float>(1, 0);
	Rt[4] = answer.at<float>(1, 1);
	Rt[5] = answer.at<float>(1, 2);
	Rt[6] = answer.at<float>(2, 0);
	Rt[7] = answer.at<float>(2, 1);
	Rt[8] = answer.at<float>(2, 2);

	Rt[9] = answer.at<float>(0,3);
	Rt[10] = answer.at<float>(1,3);
	Rt[11] = answer.at<float>(2,3);

	return Rt;
}

void Test() {

	std::string folder_left = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_0/";
	std::string folder_right = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_1";
	fs::directory_iterator left_iterator(folder_left);
	fs::directory_iterator right_iterator(folder_right);
	fs::directory_iterator left_iterator_(folder_left);
	++left_iterator_;

	std::vector<std::vector<KeyPoint>> kps_vec = NoName(left_iterator, left_iterator_, 8);

	float P0[] = { 718.856f, 0.0f, 607.1928f, 0.0f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	float P1[] = { 718.856f, 0.0f, 607.1928f, -386.1448f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	Mat P_left(3, 4, cv::DataType<float>::type, P0);
	Mat P_right(3, 4, cv::DataType<float>::type, P1);
	Mat P = Mat::eye(4, 4, CV_32F);
	P.convertTo(P, CV_32F);
	std::vector<double*> T_i_vec;
	for (int i = 0; i < kps_vec.size(); ++i) {

		left_iterator_++;

		ForOptimize buffer = EstimateMotion(imread((*left_iterator).path().u8string()),
			imread((*right_iterator).path().u8string()),
			imread((*left_iterator_).path().u8string()), P_left, P_right);
		Mat rot;
		Rodrigues(buffer.R, rot);
		rot.convertTo(rot, CV_32F);
		buffer.t.convertTo(buffer.t, CV_32F);
		Mat L = T(rot, buffer.t);
		L = L.inv();
		P *= L;
		P.convertTo(P, CV_32F);
		T_i_vec.push_back(Transform(P));
		left_iterator++;
		right_iterator++;
	}

	std::string l = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_0/000000.png";
	std::string r = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_1/000000.png";
	cv::Mat imgl = cv::imread(l);
	cv::Mat imgr = cv::imread(r);

	std::vector<double*> pts3d_vec = CalculateDepth(imgl, imgr, kps_vec, P_left, P_right);
	ceres::Problem problem;
	std::cout << "BEFORE\n";
	for (auto& T : T_i_vec)
		std::cout << T[9] << " " << T[10] << " " << T[11] << '\n';
	for(int i = 0; i < T_i_vec.size(); ++i)
		for (int j = 0; j < pts3d_vec.size(); ++j)
		{
			ceres::CostFunction* cost_function =
				SnavelyReprojectionErrorFixedPoints::Create(
					double(kps_vec[0][j].pt.x), double(kps_vec[0][j].pt.y), pts3d_vec[j][0], pts3d_vec[j][1],
					pts3d_vec[j][2], P0[0], P0[1], P0[3], P0[7]);

			problem.AddResidualBlock(cost_function, nullptr, T_i_vec[i]);
		}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << "After\n";
	for (auto& T : T_i_vec)
		std::cout << T[9] << " " << T[10] << " " << T[11] << '\n';
	//std::cout << summary.FullReport() << "\n";
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
	cv::Mat img_l = cv::imread(l);
	cv::Mat img_r = cv::imread(r);
	cv::Mat next_img = cv::imread(nl);
	//alignImages(img_l, next_img);
	ForOptimize answer = EstimateMotion(img_l, img_r, next_img, P_left, P_right);
	//Display(answer.t);
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
	fs::directory_iterator left_iterator_(folder_left);
	++left_iterator_;
	std::vector<std::vector<KeyPoint>> result = NoName(left_iterator, left_iterator_, 10);
	Test();
	/*
	for (int i = 0; i < N; ++i) {

		left_iterator_++;

		ForOptimize buffer = EstimateMotion(imread((*left_iterator).path().u8string()),
			imread((*right_iterator).path().u8string()),
			imread((*left_iterator_).path().u8string()), P_left, P_right);

		std::cout << "left" << *left_iterator << std::endl;
		std::cout << "right" << *right_iterator << std::endl;
		std::cout << "left_next" << *left_iterator_ << std::endl;
		std::cout << "-----------------\n";
		Mat rot;
		Rodrigues(buffer.R, rot);
		rot.convertTo(rot, CV_32F);
		buffer.t.convertTo(buffer.t, CV_32F);
		Mat L = T(rot, buffer.t);
		L = L.inv();
		P *= L;
		P.convertTo(P, CV_32F);
		T_i.push_back(P);

		left_iterator++;
		right_iterator++;
	}*/

	/*std::vector<ForOptimize> answers;
	std::vector<Mat> distance;
	ofstream in("C:/Users/Andrey/Desktop/Data/results.txt");
	Mat P = Mat::eye(4, 4, CV_32F);
	P.convertTo(P, CV_32F);
	const int N = 200;
	
	for (int i = 0; i < N; ++i) {

		left_iterator_++;

		ForOptimize buffer = EstimateMotion(imread((*left_iterator).path().u8string()),
			imread((*right_iterator).path().u8string()),
			imread((*left_iterator_).path().u8string()), P_left, P_right);

		Mat rot;
		Rodrigues(buffer.R, rot);
		rot.convertTo(rot, CV_32F);
		buffer.t.convertTo(buffer.t, CV_32F);
		Mat L = T(rot, buffer.t);
		L = L.inv();
		P *= L;
		P.convertTo(P, CV_32F);
		in << P.at<float>(0,3) << " " << P.at<float>(1,3) << " " << P.at<float>(2,3) << std::endl;
		left_iterator++;
		right_iterator++;
		if (i % 100 == 0)
			std::cout << i << std::endl;
	}
	
	std::vector<double*> output = Transform(T_i); // 0-68
	std::vector<std::vector<KeyPoint>> answer_ = NoName(left_iterator, left_iterator_, 20);
	std::cout << " Noname\n";// 0-68
	std::vector<double*> beta = CalculateDepth(img_l, img_r, answer_,P_left,P_right); // 0 -19
	waitKey(0);
	std::vector<double*> t_predicted;
	ceres::Problem problem;
	std::cout << beta.size() << " " << answer_.size() << " " << output.size();
	/*
	for (int i = 0; i < answer.pts_2d.size(); ++i) {
		ceres::CostFunction* cost_function =
			SnavelyReprojectionError::Create(
				double(answer.pts_2d[i].x), double(answer.pts_2d[i].y), P0[0], P0[1], P0[3], P0[7]);

		double Rt[12];
		double p3d[3];
		Mat R;
		Rodrigues(answer.R, R, noArray());
		Rt[0] = answer.R.at<float>(0, 0);
		Rt[1] = answer.R.at<float>(0, 1);
		Rt[2] = answer.R.at<float>(0, 2);
		Rt[3] = answer.R.at<float>(1, 0);
		Rt[4] = answer.R.at<float>(1, 1);
		Rt[5] = answer.R.at<float>(1, 2);
		Rt[6] = answer.R.at<float>(2, 0);
		Rt[7] = answer.R.at<float>(2, 1);
		Rt[8] = answer.R.at<float>(2, 2);
		
		Rt[9] =  answer.t.at<float>(0);
		Rt[10] = answer.t.at<float>(1);
		Rt[11] = answer.t.at<float>(2);

		p3d[0] = answer.pts_3d[i].x;
		p3d[1] = answer.pts_3d[i].y;
		p3d[2] = answer.pts_3d[i].z;
		
		t_predicted.push_back(Rt);
		problem.AddResidualBlock(cost_function, nullptr, Rt, p3d);
	}
		
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";*/
		
	return 0;
}
