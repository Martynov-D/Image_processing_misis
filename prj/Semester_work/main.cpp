#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/ximgproc.hpp"
#include <iostream>


cv::Mat getImage(std::string name)
{
    cv::Mat img = cv::imread("C:/Admin/Programming/C++/OpenCV/data/test_data/" + name);
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, CV_BGR2GRAY);

    return imgGray;
}

cv::Mat stereoBm(const cv::Mat& left, const cv::Mat& right, int nDisparities, int SADWindowSize, bool algo)
{
    cv::Mat left_disp(left.rows, left.cols, CV_32F);
    cv::Mat right_disp(left.rows, left.cols, CV_32F);
    cv::Mat disp(left.rows, left.cols, CV_32F);
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;

    // StereoBM
    if (algo)
    {
        cv::Ptr<cv::StereoBM> left_sbm = cv::StereoBM::create(nDisparities, SADWindowSize);
        //wls_filter = cv::ximgproc::createDisparityWLSFilter(left_sbm);
        //cv::Ptr<cv::StereoMatcher> right_sbm = cv::ximgproc::createRightMatcher(left_sbm);
        left_sbm->compute(left, right, left_disp);
        //right_sbm->compute(left, right, right_disp);
    }
    // StereoSGBM
    else
    {
        int 	minDisparity = 0;
        int 	numDisparities = nDisparities;
        int 	blockSize = SADWindowSize;
        int 	P1 = 8 * 3 * blockSize * blockSize;
        int 	P2 = 32 * 3 * blockSize * blockSize;
        int 	disp12MaxDiff = 0;
        int 	preFilterCap = 63;
        int 	uniquenessRatio = 0;
        int 	speckleWindowSize = 0;
        int 	speckleRange = 0;
        int 	mode = cv::StereoSGBM::MODE_HH;

        cv::Ptr<cv::StereoSGBM> left_sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize);
        left_sgbm->setP1(P1);
        left_sgbm->setP2(P2);
        left_sgbm->setPreFilterCap(preFilterCap);
        left_sgbm->setMode(mode);
        //wls_filter = cv::ximgproc::createDisparityWLSFilter(left_sgbm);
        //cv::Ptr<cv::StereoMatcher> right_sgbm = cv::ximgproc::createRightMatcher(left_sgbm);

        left_sgbm->compute(left, right, left_disp);
        //right_sgbm->compute(left, right, right_disp);
    }
    /*wls_filter->setLambda(8000);
    wls_filter->setSigmaColor(1.5);
    wls_filter->filter(left_disp, left, disp, right_disp);*/

    return left_disp;
}

int main()
{
   /* std::string left_im{ "ambush_5_left.jpg" };
    std::string right_im{ "ambush_5_right.jpg" };*/

   /* std::string left_im{ "left.png" };
    std::string right_im{ "right.png" };*/

    /*std::string left_im{ "inLeft.pgm" };
   std::string right_im{ "inRight.pgm" };*/

    std::string left_im{ "38_l.png" };
    std::string right_im{ "38_r.png" };
   
    cv::Mat left = getImage(left_im);
    cv::Mat right = getImage(right_im);

   
    int numDisparities = 16 * 4;
    int SADWindowSize = 3;
    bool mode = false;
    if (mode)
    {
        SADWindowSize = 9;
    }

    cv::Mat disp;
    cv::Mat disp8;

    disp = stereoBm(left, right, numDisparities, SADWindowSize, mode);
    cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    /*
    depth = baseline * focal / disparity
    Для KITTI  baseline = 0.54m, фокусное расстояние ~721 pix.
    Диспаратность, которая получается на выходе, должна быть преобразована к размеру исходного изображения == 1242.
    */
    cv::Mat depth = 721 * 0.54 / disp8;
    cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);

   /* cv::Mat canny;
    cv::Mat detected_edges;
    cv::Canny(disp8, detected_edges, 50, 50 * 3, 3);
    disp8.copyTo(canny, detected_edges);*/

   /* cv::Mat disp8E;
    cv::Mat filterD = (cv::Mat_<uchar>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::erode(disp8, disp8E, filterD);
    cv::Mat filterE = (cv::Mat_<uchar>(3, 3) << 0, 0, 1, 0, 1, 0, 1, 0, 0);
    cv::erode(disp8E, disp8E, filterE);*/
   /* cv::Mat depthE;
    cv::Mat filterC = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 0, 1, 0, 1, 0);
    cv::dilate(depth, depthE, filterC);*/

    cv::Mat bgrDisp;
    cv::Mat bgrDepth;
    cv::applyColorMap(disp8, bgrDisp, cv::COLORMAP_JET);
    cv::applyColorMap(depth, bgrDepth, cv::COLORMAP_JET);

    cv::imshow("Disparity map", disp8);
    cv::imshow("Depth", depth);
    cv::imshow("Colored disp", bgrDisp);
    cv::imshow("Colored depth", bgrDepth);
    
    //cv::imshow("Canny", canny);
    //cv::imshow("Erode", depthE);
  
    if (mode)
    {
        cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/disp8_BM.png", disp8, { CV_IMWRITE_PNG_COMPRESSION, 0 });
        cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/depth_BM.png", depth, { CV_IMWRITE_PNG_COMPRESSION, 0 });
        cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/disp8_BGR_BM.png", bgrDisp, { CV_IMWRITE_PNG_COMPRESSION, 0 });
        cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/depth_BGR_BM.png", bgrDepth, { CV_IMWRITE_PNG_COMPRESSION, 0 });
    }
    else
    {
        cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/disp8_SGBM.png", disp8, { CV_IMWRITE_PNG_COMPRESSION, 0 });
        cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/depth_SGBM.png", depth, { CV_IMWRITE_PNG_COMPRESSION, 0 });
        cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/disp8_BGR_SGBM.png", bgrDisp, { CV_IMWRITE_PNG_COMPRESSION, 0 });
        cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/depth_BGR_SGBM.png", bgrDepth, { CV_IMWRITE_PNG_COMPRESSION, 0 });

    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}