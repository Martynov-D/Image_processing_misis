#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

void tempTask()
{
	// READ RGB color image and convert it to Lab
	cv::Mat bgr_image = cv::imread("C:/Users/dimam/Documents/Admin/Programming/C++/OpenCV/OpenCVData/test_data/parrot.jpg");
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
}

void second()
{
	cv::Mat image = cv::imread("C:/Users/dimam/Documents/Admin/Programming/C++/OpenCV/OpenCVData/test_data/parrot.jpg", CV_LOAD_IMAGE_COLOR);

	// Преобразование в изображение в оттенках серого
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);

	// Повышение яркости изображения
	cv::Mat imageB = image + cv::Scalar(30, 30, 30);

	cv::Mat imageGrayB;
	cv::cvtColor(imageB, imageGrayB, CV_BGR2GRAY);

	int histSize{ 256 };
	float range[] = { 0,256 };
	const float* histRange = { range };

	bool uniform{ true };
	bool accumulate{ false };

	int hist_w{ 512 }; // 512
	int hist_h{ 400 }; // 400
	int bin_w{ cvRound((double)hist_w / histSize) };

	// Создание пустых изображений для гистограмм
	cv::Mat hist_window(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat histBright_window(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

	// Рассчет гистограмм
	cv::Mat hist;
	cv::Mat hist_bright;
	calcHist(&imageGray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&imageGrayB, 1, 0, cv::Mat(), hist_bright, 1, &histSize, &histRange, uniform, accumulate);

	// Нормализация гистограмм под размер окна
	cv::normalize(hist, hist, 0, hist_window.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(hist_bright, hist_bright, 0, histBright_window.rows, cv::NORM_MINMAX, -1, cv::Mat());

	// Построение гистограмм
	for (int i = 1; i < histSize; ++i)
	{
		line(hist_window, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
		line(histBright_window, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_bright.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist_bright.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}

	// Отображение результатов

	cv::namedWindow("Original image in gray", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Hist", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Bright image", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Bright Hist", cv::WINDOW_AUTOSIZE);

	cv::imshow("Original image in gray", imageGray);
	cv::imshow("Hist", hist_window);
	cv::imshow("Bright image", imageGrayB);
	cv::imshow("Bright Hist", histBright_window);

	cv::waitKey(0);

	cv::destroyAllWindows();
}


int main()
{
	//tempTask();
	//second();

	return 0;
}