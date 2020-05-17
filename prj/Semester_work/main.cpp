#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat getImage(std::string name)
{
    cv::Mat img = cv::imread("C:/Admin/Programming/C++/OpenCV/data/test_data/" + name);
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, CV_BGR2GRAY);

    return imgGray;
}

cv::Mat sgbm(const cv::Mat& left, const cv::Mat& right, int nDisparities, int SADWindowSize)
{
    int 	minDisparity = 0;
    int 	numDisparities = nDisparities;
    int 	blockSize = SADWindowSize;
    int 	P1 = 8 * 1 * blockSize * blockSize;
    int 	P2 = 32 * 1 * blockSize * blockSize;
    int 	disp12MaxDiff = 10;
    int 	preFilterCap = 4;
    int 	uniquenessRatio = 1;
    int 	speckleWindowSize = 150;
    int 	speckleRange = 2;
    int 	mode = cv::StereoSGBM::MODE_SGBM;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);
    cv::Mat disp;
    sgbm->compute(left, right, disp);

    return disp;
}

cv::Mat bm(const cv::Mat& left, const cv::Mat& right, int nDisparities, int SADWindowSize)
{
    cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(nDisparities, SADWindowSize);
    cv::Mat disp;
    sbm->compute(left, right, disp);

    return disp;
}

int main()
{
    /*std::string left_im{ "ambush_5_left.jpg" };
    std::string right_im{ "ambush_5_right.jpg" };*/

   /* std::string left_im{ "left.png" };
    std::string right_im{ "right.png" };*/

    std::string left_im{ "5_l.png" };
    std::string right_im{ "5_r.png" };

    cv::Mat left = getImage(left_im);
    cv::Mat right = getImage(right_im);

    int numDisparities = 16 * 4;
    int SADWindowSize = 11;

    cv::Mat disp;
    cv::Mat disp8;
    cv::Mat disp32f;

    if (false)
        disp = bm(left, right, numDisparities, SADWindowSize);
    else
        disp = sgbm(left, right, numDisparities, SADWindowSize);
    
    /*  
    depth = baseline * focal / disparity
    Для KITTI  baseline = 0.54m, фокусное расстояние ~721 pix.
    Диспаратность, которая получается на выходе, должна быть преобразована к размеру исходного изображения == 1242.
    */
    disp.convertTo(disp32f, CV_32F);
    cv::Mat depth(disp32f.cols, disp32f.rows, disp32f.type());

    for (int i{ 0 }; i < depth.rows; ++i)
    {
        for (int j{ 0 }; j < depth.cols; ++j)
            depth.at<float>(i, j) = 127;//disp32f.at<float>(i, j);
    }

    //cv::Mat depth = 0.54 * 721 / (1242 * disp);
    cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    cv::imshow("Disparity map", disp8);
    cv::imshow("Depth", depth);

    //cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/bm.png", bmImage, { CV_IMWRITE_PNG_COMPRESSION, 0 });
    //cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/sgbm.png", sgbmImage, { CV_IMWRITE_PNG_COMPRESSION, 0 });

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}