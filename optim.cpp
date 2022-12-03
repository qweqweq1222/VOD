#include "lib.h"
# define M_PI  3.14159265358979323846
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

void Display(const Mat& mtx)
{
	for (int i = 0; i < mtx.rows; ++i)
	{
		for (int j = 0; j < mtx.cols; ++j)
			std::cout << mtx.at<float>(i, j) << " ";
		std::cout << endl;
	}
}

vector<double> GetAnglesAndVec(const Mat& Rt)
{
	double beta = asin(Rt.at<float>(0, 2));
	double alpha = acos(Rt.at<float>(2, 2) / cos(beta));
	double gamma = acos(Rt.at<float>(0, 0) / cos(beta));
	return { alpha, beta, gamma, Rt.at<float>(0,3), Rt.at<float>(1,3), Rt.at<float>(2,3) };
}
/*void EstimateAndOptimize(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight)
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
		cout << " SZIE :: " << vec.size() << endl;
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
			std::vector<double> vector = Transform_vec(P);
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
					//problem.SetParameterLowerBound(crt[i], idx, buffer - fabs(buffer) * 0.3);
					problem.SetParameterUpperBound(crt[i], idx, 1.0f);
					//problem.SetParameterUpperBound(crt[i], idx, buffer + fabs(buffer) * 0.3);
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
			for (int j = 0; j < 12; ++j)
				std::cout << crt[i][j] << " ";
			std::cout << std::endl;
			Mat copy_GLOBAL = GLOBAL_P.clone();
			copy_GLOBAL *= FromPointerToMat(crt[i]);
			in << copy_GLOBAL.at<float>(0, 3) << " " << copy_GLOBAL.at<float>(1, 3) << " " << copy_GLOBAL.at<float>(2, 3) << "\n";
			if (i == crt.size() - 1)
				GLOBAL_P *= FromPointerToMat(crt[i]);
		}

		for (auto& p : crt)
			delete[] p;
		for (auto& d : pts_3d)
			delete[] d;

	}
}*/
Mat ReconstructFromV4(double* alpha_trans)
{
	Mat answer = Mat::eye(4, 4, CV_32F);
	answer.at<float>(0, 0) = cos(alpha_trans[0]);
	answer.at<float>(0, 2) = sin(alpha_trans[0]);
	answer.at<float>(2, 0) = -sin(alpha_trans[0]);
	answer.at<float>(2, 2) = cos(alpha_trans[0]);
	answer.at<float>(0, 3) = alpha_trans[1];
	answer.at<float>(1, 3) = alpha_trans[2];
	answer.at<float>(2, 3) = alpha_trans[3];
	return answer;
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
		vector<vector<double>> location;
		fs::directory_iterator copy_left(left_path);
		fs::directory_iterator copy_next(left_path);

		std::advance(copy_next, START_KEY_FRAME + counter + 1);
		std::advance(copy_left, START_KEY_FRAME + counter);

		vec = GetSamePoints(copy_left, copy_next, SAME_POINTS);
		vector<vector<KeyPoint>> alternative(vec.size(), vector<KeyPoint>(0));
		vector<Vec3f> pts3d;
		counter += vec.size();
		buffer = vec.size();
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

				for (int p = 0; p < vec[0].size(); ++p) {
					float u = (float(vec[0][p].pt.x));
					float v = (float(vec[0][p].pt.y));
					float z = coeff / disparity.at<float>(int(v), int(u));
					if (z > 0 && z < DEPTH_TRASH)
					{
						float x = z * (v - cil.cameraMatrix.at<float>(0, 2)) / cil.cameraMatrix.at<float>(0, 0);
						float y = z * (u - cil.cameraMatrix.at<float>(1, 2)) / cil.cameraMatrix.at<float>(1, 1);
						for (int idx = 0; idx < vec.size(); ++idx)
							alternative[idx].push_back(vec[idx][p]);
						pts3d.push_back({ x,y,z });
					}

				}
			}


			/* Check if pts2d_3d shape is proper for alternative*/
			/* check alternative */
			/*
			if (TRUE)
			{
	
				if (alternative[0].size() == pts3d.size())
				{
					for (int i = 0; i < alternative[0].size(); ++i)
					{
						for (int j = 0; j < alternative.size(); ++j)
							cout << alternative[j][i].pt.x << " " << alternative[j][i].pt.y << " | ";
						cout << endl;
					}
				}
				
				double z_shift = 0.5; 
				for (auto& p : pts3d)
					p[2] -= z_shift;
				for (int i = 0; i < alternative[0].size(); ++i)
				{
					double x = PLeft.at<float>(0, 0) * pts3d[i][0] / pts3d[i][2] + PLeft.at<float>(0, 2);
					double y = PLeft.at<float>(0, 0) * pts3d[i][1] / pts3d[i][2] + PLeft.at<float>(1, 2);
					cout << "{ " << alternative[1][i].pt.y << ", " << alternative[1][i].pt.x << "}--{" << x << ", " << y << "} _";
					cout << endl;
				}

				return;
			}
			*/



			++left_iterator;
			++right_iterator;
			++next_iterator;
			Mat rot;
			cv::Rodrigues(buffer.first, rot);
			rot.convertTo(rot, CV_32F);
			buffer.second.convertTo(buffer.second, CV_32F);
			Mat L = T(rot, buffer.second);
			P *= L.inv();
			std::vector<double> a_t = {acos(P.at<float>(0,0)),P.at<float>(0, 3), P.at<float>(1, 3), P.at<float>(2, 3)};
			location.push_back(a_t);
		}
		ceres::Problem problem;
		vector<double*> alphas_trans(location.size());
		for (int i = 0; i < location.size(); ++i)
		{
			alphas_trans[i] = new double[4];
			for (int j = 0; j < 4; ++j)
				alphas_trans[i][j] = location[i][j];
		}
		std::vector<double*> pts_3d(pts3d.size());
		for (int i = 0; i < pts3d.size(); ++i)
		{
			pts_3d[i] = new double[3];
			pts_3d[i][0] = pts3d[i][0];
			pts_3d[i][1] = pts3d[i][1];
			pts_3d[i][2] = pts3d[i][2];
		}
		double buffer_mas[12];
		
		for (int i = 0; i < alternative.size() - 1; ++i)
		{
			for (int j = 0; j < pts3d.size(); ++j)
			{
			
					/* imbalance checker */

					double P3[3];
					P3[0] = cos(alphas_trans[i][0]) * (pts_3d[j][0] - alphas_trans[i][1]) - sin(alphas_trans[i][0]) * (pts_3d[j][2] - alphas_trans[i][3]);
					P3[1] = pts_3d[j][1] - alphas_trans[i][2];
					P3[2] = sin(alphas_trans[i][0]) * (pts_3d[j][0] - alphas_trans[i][1]) + cos(alphas_trans[i][0]) * (pts_3d[j][2] - alphas_trans[i][3]);
					double a = PLeft.at<float>(0, 0) * pts_3d[j][0] / pts_3d[j][2] + PLeft.at<float>(0, 2);
					double b = PLeft.at<float>(0, 0) * pts_3d[j][1] / pts_3d[j][2] + PLeft.at<float>(1, 2);
					a -= double(alternative[i + 1][j].pt.y);
					b -= double(alternative[i + 1][j].pt.x);

					if (abs(a) < 100 && abs(b) < 100)
					{
						ceres::CostFunction* cost_function =
							SnavelyReprojectionError::Create(double(alternative[i + 1][j].pt.y), double(alternative[i + 1][j].pt.x),
								PLeft.at<float>(0, 0), PLeft.at<float>(0, 0), PLeft.at<float>(0, 2), PLeft.at<float>(1, 2));
						problem.AddResidualBlock(cost_function, nullptr, alphas_trans[i], pts_3d[j]);
						problem.SetParameterLowerBound(alphas_trans[i], 0, -M_PI / 2);
						problem.SetParameterUpperBound(alphas_trans[i], 0, M_PI / 2);
						for (int index = 1; index < 4; ++index)
						{
							if (index != 2) //30% вариации 
							{
								double abs_ = abs(alphas_trans[i][index]);
								double lower_bound = alphas_trans[i][index] - 0.3 * abs_;
								double upper_bound = alphas_trans[i][index] + 0.3 * abs_;
								problem.SetParameterLowerBound(alphas_trans[i], index, lower_bound);
								problem.SetParameterUpperBound(alphas_trans[i], index, upper_bound);
							}
							else //по y (вертикали) мы почти никуда не смещаемся - 5%
							{
								double abs_ = abs(alphas_trans[i][index]);
								double lower_bound = alphas_trans[i][index] - 0.01 * abs_;
								double upper_bound = alphas_trans[i][index] + 0.01 * abs_;
								problem.SetParameterLowerBound(alphas_trans[i], index, lower_bound);
								problem.SetParameterUpperBound(alphas_trans[i], index, upper_bound);
								problem.SetParameterLowerBound(alphas_trans[i], index, -1.0);
								problem.SetParameterUpperBound(alphas_trans[i], index, 1.0);
							}
						}
					}
					//double P3[3];
					//std::cout << pts_3d[j][0] << " " << pts_3d[j][1] << " " << pts_3d[j][2] << endl;
					//std::cout << alphas_trans[i][1] << " " << alphas_trans[i][2] << " " << alphas_trans[i][3] << endl;
					//P3[0] = cos(alphas_trans[i][0]) * (pts_3d[j][0] - alphas_trans[i][1]) - sin(alphas_trans[i][0]) * (pts_3d[j][2] - alphas_trans[i][3]);
					//P3[1] = pts_3d[j][1] - alphas_trans[i][2];
					//P3[2] = sin(alphas_trans[i][0]) * (pts_3d[j][0] - alphas_trans[i][1]) + cos(alphas_trans[i][0]) * (pts_3d[j][2] - alphas_trans[i][3]);
					//std::cout << P3[0] << " " << P3[1] << " " << P3[2] << endl;
					//std::cout << P3[0]/P3[2] << " " << P3[1]/P3[2] << " " << P3[2]/P3[2] << endl;
					//std::cout << "-----------------------\n";

					//P3[0] = cos(alphas_trans[i][0]) * (pts_3d[j][0]) + sin(alphas_trans[i][0]) * (pts_3d[j][2]) + alphas_trans[i][1];
					//P3[1] = pts_3d[j][1] + alphas_trans[i][2];
					//P3[2] = -sin(alphas_trans[i][0]) * (pts_3d[j][0]) + cos(alphas_trans[i][0]) * (pts_3d[j][2]) + alphas_trans[i][3];
					//cout << PLeft.at<float>(0, 0) * pts_3d[j][0] / pts_3d[j][2] + PLeft.at<float>(0, 2) << " " << PLeft.at<float>(0, 0) * pts_3d[j][1] / pts_3d[j][2] + PLeft.at<float>(1, 2) << endl;
					//cout << vec[0][j].pt.x  << " " << vec[0][j].pt.y << endl;
					//double predicted_x = PLeft.at<float>(0, 0) * P3[0] / P3[2] + PLeft.at<float>(0, 2);
					//double predicted_y = PLeft.at<float>(0, 0) * P3[1] / P3[2] + PLeft.at<float>(1, 2);

					//cout << "{" << alternative[i + 1][j].pt.y << "," << alternative[i + 1][j].pt.x << "}{" << predicted_x << "," << predicted_y << "}\n";
					//cout << "{" << alphas_trans[i][0] << "," << alphas_trans[i][1] << "," << alphas_trans[i][2] << "," << alphas_trans[i][3] << "\n";
			}
		}
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		for (int i = 0; i < alphas_trans.size(); ++i)
		{
			Mat copy_GLOBAL = GLOBAL_P.clone();
			copy_GLOBAL *= ReconstructFromV4(alphas_trans[i]);
			in << copy_GLOBAL.at<float>(0, 3) << " " << copy_GLOBAL.at<float>(1, 3) << " " << copy_GLOBAL.at<float>(2, 3) << "\n";
			if (i == alphas_trans.size() - 1)
				GLOBAL_P *= ReconstructFromV4(alphas_trans[i]);
		}

		for (auto& p : alphas_trans)
			delete[] p;
		for (auto& d : pts_3d)
			delete[] d;

	}
}
