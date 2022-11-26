#include "lib.h"


Mat FromPointerToMat(double* pt)
{
	Mat T = Mat::eye(4, 4, CV_32F);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			T.at<float>(i, j) = pt[3 * i + j];
	T.at<float>(0, 3) = pt[9];
	T.at<float>(1, 3) = pt[10];
	T.at<float>(1, 3) = pt[11];
	return T;
}
void EstimateAndOptimize(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight) 
{
	CameraInfo cil = Decompose(PLeft);
	CameraInfo cir = Decompose(PRight);
	fs::directory_iterator left_iterator(left_path);
	fs::directory_iterator right_iterator(right_path);
	fs::directory_iterator next_iterator(left_path);

	std::advance(left_iterator, START_KEY_FRAME);
	std::advance(right_iterator, START_KEY_FRAME);
	std::advance(next_iterator, START_KEY_FRAME + 1);
	std::vector<std::vector<KeyPoint>> vec;
	std::ofstream in(input);
	int buffer = 0;
	int counter = 0;
	Mat GLOBAL_P = Mat::eye(4, 4, CV_32F);

	for (int i = 0; i < NUM_OF_FRAMES; i += buffer)
	{
		std::vector<std::vector<double>> T_i;
		fs::directory_iterator copy_left(left_path);
		fs::directory_iterator copy_next(left_path);

		std::advance(copy_next, START_KEY_FRAME + counter + 1);
		std::advance(copy_left, START_KEY_FRAME + counter);

		vec = GetSamePoints(copy_left, copy_next, SAME_POINTS);
		counter += vec.size();
		buffer = vec.size();
		std::vector<Point3f> pts3d;

		Mat P = Mat::eye(4, 4, CV_32F);
		for (int jdx = 0; jdx < vec.size() - 1; ++jdx)
		{

			std::pair<Mat, Mat> buffer = EstimateMotion(imread((*left_iterator).path().u8string()),
				imread((*right_iterator).path().u8string()),
				imread((*next_iterator).path().u8string()), PLeft, PRight);
			if (jdx == 0)
			{
				Mat disparity = CalculateDisparity(imread((*left_iterator).path().u8string()), imread((*next_iterator).path().u8string()));
				float coeff = cil.cameraMatrix.at<float>(0, 0) * (cir.transVector.at<float>(0) - cil.transVector.at<float>(0));

				for (auto& pt : vec[0]) {
					float u = (float(pt.pt.x));
					float v = (float(pt.pt.y));
					float z = coeff / disparity.at<float>(int(v), int(u));
					if (z > 0 && z < DEPTH_TRASH)
					{
						float x = z * (u - cil.cameraMatrix.at<float>(0, 2)) / cil.cameraMatrix.at<float>(0, 0);
						float y = z * (v - cil.cameraMatrix.at<float>(1, 2)) / cil.cameraMatrix.at<float>(1, 1);
						pts3d.push_back({ x, y, z });
					}
				}
			}

			++left_iterator;
			++right_iterator;
			++next_iterator;

			Mat rot;
			cv::Rodrigues(buffer.first, rot);
			rot.convertTo(rot, CV_32F);
			buffer.second.convertTo(buffer.second, CV_32F);
			Mat L = T(rot, buffer.second);
			P *= L.inv();
			std::vector<double> vector = Transform_vec(P.inv());
			T_i.push_back(vector);
		}
		ceres::Problem problem;
		std::vector<double*> crt(T_i.size());
		for (int i = 0; i < T_i.size(); ++i)
		{
			crt[i] = new double[12];
			for (int j = 0; j < 12; ++j)
				crt[i][j] = double(T_i[i][j]);
		}
		std::vector<double*> pts_3d(pts3d.size());
		for (int i = 0; i < pts3d.size(); ++i)
		{
			pts_3d[i] = new double[3];
			pts_3d[i][0] = double(pts3d[i].x);
			pts_3d[i][1] = double(pts3d[i].y);
			pts_3d[i][2] = double(pts3d[i].z);
		}
		double buffer_mas[12];
		for (int i = 0; i < vec.size() - 1; ++i)
		{
			for (int j = 0; j < pts3d.size(); ++j)
			{
				ceres::CostFunction* cost_function =
					SnavelyReprojectionError::Create(double(vec[i + 1][j].pt.x), double(vec[i + 1][j].pt.y),
						PLeft.at<float>(0,0), PLeft.at<float>(0, 0), PLeft.at<float>(0, 2), PLeft.at<float>(1, 2));
				problem.AddResidualBlock(cost_function, nullptr, crt[i], pts_3d[j]);
				for (int idx = 0; idx < 9; ++idx)
				{
					double buffer = crt[i][idx];
					problem.SetParameterLowerBound(crt[i], idx, -1.0f);
					problem.SetParameterLowerBound(crt[i], idx, buffer - fabs(buffer) * 0.3);
					problem.SetParameterUpperBound(crt[i], idx, 1.0f);
					problem.SetParameterUpperBound(crt[i], idx, buffer + fabs(buffer) * 0.3);
				}
				for (int idx = 9; idx < 12; ++idx)
				{
					double buffer = crt[i][idx];
					if (idx != 11)
					{
						problem.SetParameterLowerBound(crt[i], idx, buffer - fabs(buffer) * 0.3);
						problem.SetParameterUpperBound(crt[i], idx, buffer + fabs(buffer) * 0.3);
					}
					else
					{
						problem.SetParameterLowerBound(crt[i], idx, 0.8 * buffer);
						problem.SetParameterUpperBound(crt[i], idx, buffer * 1.2);
					}
				}
			}
		}

		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		for (int i = 0; i < crt.size(); ++i)
		{
			Mat copy_GLOBAL = GLOBAL_P.clone();
			copy_GLOBAL *= FromPointerToMat(crt[i]).inv();
			in << copy_GLOBAL.at<float>(0, 3) << " " << copy_GLOBAL.at<float>(1, 3) << " " << copy_GLOBAL.at<float>(2, 3) << "\n";
			if (i = crt.size() - 1)
				GLOBAL_P *= FromPointerToMat(crt[i]).inv();
		}

		for (auto& p : crt)
			delete[] p;
		for (auto& d : pts_3d)
			delete[] d;

	}
}
