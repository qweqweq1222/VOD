#include "lib.h"

KeyPointMatches AlignImages(Mat& im1, Mat& im2) {
	Mat im1Gray, im2Gray, descriptors1, descriptors2;
	cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);
	vector<KeyPoint> keypoints1, keypoints2;
	std::vector< std::vector<DMatch> > knn_matches;
	Ptr<SIFT> detector = SIFT::create();
	detector->detectAndCompute(im1Gray, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(im2Gray, noArray(), keypoints2, descriptors2);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	const float ratio_thresh = 0.7f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			good_matches.push_back(knn_matches[i][0]);
	std::sort(good_matches.begin(), good_matches.end());
	if (good_matches.size() * GOOD_MATCH_PERCENT >= MIN_FEATURES)
		good_matches.erase(good_matches.begin() + good_matches.size() * GOOD_MATCH_PERCENT, good_matches.end());
	return KeyPointMatches(good_matches, keypoints1, keypoints2);;
}
Mat CalculateDisparity(const cv::Mat& left_image, const cv::Mat& right_image) {
	Mat l, r, disparity;
	int sad_window = 6;
	int num_disparities = sad_window * 16;
	int block_size = 11;

	cvtColor(left_image, l, cv::COLOR_BGR2GRAY);
	cvtColor(right_image, r, cv::COLOR_BGR2GRAY);
	Ptr<cv::StereoSGBM> stereo = StereoSGBM::create(0, num_disparities, block_size, 864, 3456, 0, 0, 0, 0, 0, 2);
	stereo->compute(l, r, disparity);
	disparity.convertTo(disparity, CV_32F, 1.0 / 16.0);
	return disparity;
}
CameraInfo Decompose(Mat proj_matrix)
{
	Mat K(3, 3, cv::DataType<float>::type); // intrinsic parameter matrix
	Mat R(3, 3, cv::DataType<float>::type); // rotation matrix
	Mat T(4, 1, cv::DataType<float>::type);
	decomposeProjectionMatrix(proj_matrix, K, R, T);
	return { K, R, T };
}
pair<Mat, Mat> EstimateMotion(Mat left, Mat right, Mat next, Mat P_left, Mat P_right) {

	vector<Point3f> object_points;
	vector<Point2f> image_points;
	CameraInfo cil = Decompose(P_left);
	CameraInfo cir = Decompose(P_right);
	KeyPointMatches kpm = AlignImages(left, next);
	Mat show;
	Mat disparity = CalculateDisparity(left, right);
	float coeff = cil.cameraMatrix.at<float>(0, 0) * (cir.transVector.at<float>(0) - cil.transVector.at<float>(0));
	std::vector<DMatch> good_matches;
	for (auto& match : kpm.matches) {

		float u = (float(kpm.kp1.at(match.queryIdx).pt.x));
		float v = (float(kpm.kp1.at(match.queryIdx).pt.y));
		float z = coeff / disparity.at<float>(int(v), int(u));
		if (z < DEPTH_TRASH && z > 0) {
			good_matches.push_back(match);
			Point2f pt_2d(kpm.kp2.at(match.trainIdx).pt.x, kpm.kp2.at(match.trainIdx).pt.y);
			float x = z * (u - cil.cameraMatrix.at<float>(0, 2)) / cil.cameraMatrix.at<float>(0, 0);
			float y = z * (v - cil.cameraMatrix.at<float>(1, 2)) / cil.cameraMatrix.at<float>(1, 1);
			object_points.emplace_back(Point3f{ x, y, z });
			image_points.emplace_back(pt_2d);
		}
	}

	Mat R, t;
	solvePnPRansac(object_points, image_points, cil.cameraMatrix, noArray(), R, t);
	return { R, t };
}
Mat TAP(const Mat& R, const Mat& t)
{
	Mat P = Mat::eye(4, 4, CV_32F);
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			P.at<float>(i, j) = R.at<float>(i, j);
		P.at<float>(i, 3) = t.at<float>(i);
	}
	return P;
}
std::vector<std::vector<KeyPoint>> GetSamePoints(fs::directory_iterator left, fs::directory_iterator next, const int& N_features)
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

	++left;
	++next;

	while (final_res[0].size() >= N_features)
	{
		l = imread((*left).path().u8string());
		n = imread((*next).path().u8string());

		KeyPointMatches buffer = AlignImages(l, n);
		std::vector<KeyPoint> adder;
		std::vector<KeyPoint> copy = final_res[final_res.size() - 1];
		for (auto& p : copy) {

			auto it = std::find_if(begin(buffer.matches), end(buffer.matches),
				[&](DMatch m) { return buffer.kp1.at(m.queryIdx).pt.x  == p.pt.x && buffer.kp1.at(m.queryIdx).pt.y == p.pt.y; });

			auto it_ = std::find_if(final_res[final_res.size() - 1].begin(), final_res[final_res.size() - 1].end(),
				[&](KeyPoint P) { return P.pt.x == p.pt.x && P.pt.y == p.pt.y; });

			int idx = it_ - final_res[final_res.size() - 1].begin();

			if (it != buffer.matches.end())
				adder.push_back(buffer.kp2.at((*it).trainIdx));
			else
				for (auto& el : final_res)
					el.erase(el.begin() + idx);

			if (final_res[0].size() == N_features)
				break;
		}

		if (final_res[0].size() == N_features)
			break;
		else
		{
			++left;
			++next;
			final_res.push_back(adder);
		}
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
std::vector<double> Transform_vec(const Mat answer)
{
	std::vector<double> Rt(12);

	Rt[0] = answer.at<float>(0, 0);
	Rt[1] = answer.at<float>(0, 1);
	Rt[2] = answer.at<float>(0, 2);
	Rt[3] = answer.at<float>(1, 0);
	Rt[4] = answer.at<float>(1, 1);
	Rt[5] = answer.at<float>(1, 2);
	Rt[6] = answer.at<float>(2, 0);
	Rt[7] = answer.at<float>(2, 1);
	Rt[8] = answer.at<float>(2, 2);

	Rt[9] = answer.at<float>(0, 3);
	Rt[10] = answer.at<float>(1, 3);
	Rt[11] = answer.at<float>(2, 3);
	return Rt;
}
void VisualOdometry(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight)
{


	Mat GLOBAL_P = Mat::eye(4, 4, CV_32F);

	fs::directory_iterator left_iterator(left_path);
	fs::directory_iterator right_iterator(right_path);
	fs::directory_iterator next_iterator(left_path);

	advance(next_iterator, START_KEY_FRAME);
	advance(next_iterator, START_KEY_FRAME);
	advance(next_iterator, START_KEY_FRAME + 1);

	std::ofstream in(input);

	for (int i = 0; i < NUM_OF_FRAMES; ++i)
	{
		pair<Mat, Mat> TR = EstimateMotion(imread((*left_iterator).path().u8string()),
			imread((*right_iterator).path().u8string()),
			imread((*next_iterator).path().u8string()), PLeft, PRight);
		TR.first.convertTo(TR.first, CV_32F);
		TR.second.convertTo(TR.second, CV_32F);
		Mat R, t;
		Rodrigues(TR.first, R);
		R.convertTo(R, CV_32F);
		Mat Trap = TAP(R, TR.second);
		Trap = Trap.inv();
		GLOBAL_P *= Trap;
		++left_iterator;
		++right_iterator;
		++next_iterator;
		std::cout << *left_iterator << std::endl;
		std::cout << *right_iterator << std::endl;
		std::cout << *next_iterator << std::endl << "___________________________________\n";
		in << GLOBAL_P.at<float>(0, 3) << " " << GLOBAL_P.at<float>(1, 3) << " " << GLOBAL_P.at<float>(2, 3) << std::endl;
	}
}