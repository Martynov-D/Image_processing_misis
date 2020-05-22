#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

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
    int 	disp12MaxDiff = 0;
    int 	preFilterCap = 0;
    int 	uniquenessRatio = 0;
    int 	speckleWindowSize = 0;
    int 	speckleRange = 0;
    int 	mode = cv::StereoSGBM::MODE_HH;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);
    cv::Mat disp(left.rows, left.cols, CV_32FC1);
    sgbm->compute(left, right, disp);

    return disp;
}

cv::Mat bm(const cv::Mat& left, const cv::Mat& right, int nDisparities, int SADWindowSize)
{
    cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(nDisparities, SADWindowSize);
    cv::Mat disp(left.rows, left.cols, CV_32FC1);
    sbm->compute(left, right, disp);

    return disp;
}

int main()
{
   /* std::string left_im{ "ambush_5_left.jpg" };
    std::string right_im{ "ambush_5_right.jpg" };*/

   /* std::string left_im{ "left.png" };
    std::string right_im{ "right.png" };*/

    std::string left_im{ "38_l.png" };
    std::string right_im{ "38_r.png" };
    
    /*std::string left_im{ "inLeft.pgm" };
    std::string right_im{ "inRight.pgm" };*/

    cv::Mat left = getImage(left_im);
    cv::Mat right = getImage(right_im);

    int numDisparities = 16 * 4;
    int SADWindowSize = 9;

    cv::Mat disp;
    cv::Mat disp8;

    if (false)
        disp = bm(left, right, numDisparities, SADWindowSize);
    else
        disp = sgbm(left, right, numDisparities, SADWindowSize);
    cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    /*
    depth = baseline * focal / disparity
    Для KITTI  baseline = 0.54m, фокусное расстояние ~721 pix.
    Диспаратность, которая получается на выходе, должна быть преобразована к размеру исходного изображения == 1242.
    */
    /*cv::Mat depth = 721 * 0.54 / disp8;
    cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);*/

    /*cv::Mat dst;
    cv::Mat detected_edges;
    cv::Canny(disp8, detected_edges, 50, 50 * 3, 3);
    disp8.copyTo(dst, detected_edges);*/

   /* cv::Mat disp8E;
    cv::Mat filterD = (cv::Mat_<uchar>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::erode(disp8, disp8E, filterD);
    cv::Mat filterE = (cv::Mat_<uchar>(3, 3) << 0, 0, 1, 0, 1, 0, 1, 0, 0);
    cv::erode(disp8E, disp8E, filterE);*/
   /* cv::Mat depthE;
    cv::Mat filterC = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 0, 1, 0, 1, 0);
    cv::dilate(depth, depthE, filterC);*/

    cv::Mat bgr;
    cv::applyColorMap(disp8, bgr, cv::COLORMAP_JET);

    cv::imshow("Disparity map", disp8);
    //cv::imshow("Depth", depth);
    cv::imshow("Colored", bgr);
    //cv::imshow("Canny", dst);
    //cv::imshow("Erode", depthE);
  

    //cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/disp8.png", disp8, { CV_IMWRITE_PNG_COMPRESSION, 0 });
    //cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/invertedDisp.png", depth, { CV_IMWRITE_PNG_COMPRESSION, 0 });

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}