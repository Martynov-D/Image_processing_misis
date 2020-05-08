#include <opencv2/opencv.hpp>
#include <vector>

void grayAndColorHist(cv::Mat image)
{
	// Преобразование в изображение в оттенках серого
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);

	// Вектор, который хранит разложение на каналы
	std::vector<cv::Mat> bgrPlanes;
	cv::split(image, bgrPlanes);

	int histSize{ 256 };
	float range[] = { 0,256 };
	const float* histRange = { range };

	bool uniform{ true };
	bool accumulate{ false };

	cv::Mat b_hist;
	cv::Mat g_hist;
	cv::Mat r_hist;

	cv::Mat gray_hist;

	// Рассчет гистограмм
	calcHist(&bgrPlanes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgrPlanes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgrPlanes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	calcHist(&imageGray, 1, 0, cv::Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);

	int hist_w{ 512 }; // 512
	int hist_h{ 400 }; // 400
	int bin_w{ cvRound((double)hist_w / histSize) };

	// Создание пустых изображений для гистограмм
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	cv::Mat grayHistImage(hist_h, hist_w, CV_8UC1, 255);

	// Нормализация гистограмм под размер окна
	cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	cv::normalize(gray_hist, gray_hist, 0, grayHistImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	// Построение гистограмм
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);

		line(grayHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(gray_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(gray_hist.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}

	/// Отображение результатов
	cv::namedWindow("Original image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Original image", image);

	cv::namedWindow("Hist", CV_WINDOW_AUTOSIZE);
	cv::imshow("Hist", histImage);

	cv::namedWindow("Original image in gray", CV_WINDOW_AUTOSIZE);
	cv::imshow("Original image in gray", imageGray);

	cv::namedWindow("Gray Hist", CV_WINDOW_AUTOSIZE);
	cv::imshow("Gray Hist", grayHistImage);

	cv::waitKey(0);
}

int main()
{
	// READ RGB color image and convert it to Lab
	cv::Mat bgr_image = cv::imread("C:/Admin/Programming/C++/OpenCV/data/test_data/parrot.jpg");
	cv::Mat lab_image;
	cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

	// Extract the L channel
	std::vector<cv::Mat> lab_planes(3);
	cv::split(lab_image, lab_planes); // now we have the L image in lab_planes[0]

	// apply the CLAHE algorithm to the L channel
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	cv::Mat dst;
	clahe->apply(lab_planes[0], dst);

	// Merge the the color planes back into an Lab image
	dst.copyTo(lab_planes[0]);
	cv::merge(lab_planes, lab_image);

	// convert back to RGB
	cv::Mat image_clahe;
	cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

	// display the results (you might also want to see lab_planes[0] before and after).
	cv::imshow("image original", bgr_image);
	cv::imshow("image CLAHE", image_clahe);
	cv::waitKey();
	cv::destroyAllWindows();

	return 0;
}