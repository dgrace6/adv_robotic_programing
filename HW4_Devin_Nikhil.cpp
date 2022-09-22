#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace cv::xfeatures2d;

using std::cout;
using std::endl;


int main(){

    // Establising some values to be used
    int down_width = 1500;
    int down_height = 1400;
    Mat resized_down;

    // Grab the video file
    VideoCapture cap("../required_files/IMG_6826.MOV");


    // If cap is unable to open anything, terminate the program
    if(!cap.isOpened()){

        cout << "Error opening video stream or file" << endl;

    return -1;
    }

    while(1){
        
        // Using the Mat class to store the frame taken from the video
        Mat frame;
        
        // Capture frame-by-frame
        cap >> frame;


        // Break the loop if there is no frames left
        if (frame.empty())
            break;
       
        // Using the Mat class to create objects for the frame and image
        Mat img_object;
        Mat img_scene;

        // Changing frame to gray and storing it in img_scene
        cvtColor(frame, img_scene, CV_RGB2GRAY); 

        // Creating an object to read in an image and turn it gray
        img_object = imread("../required_files/Book.png", IMREAD_GRAYSCALE);

        // If either do not exist
        if ( img_object.empty() || img_scene.empty() ) {

            cout << "Could not open or find the image!\n" << endl;
            return -1;

        }

        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        int minHessian = 400;
        Ptr<SURF> detector = SURF::create( minHessian );

        std::vector<KeyPoint> keypoints_object, keypoints_scene;
        Mat descriptors_object, descriptors_scene;


        detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
        detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );


        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used

        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
 

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.75f;
        std::vector<DMatch> good_matches;
        for ( size_t i = 0; i < knn_matches.size(); i++ ) {
            if ( knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance ) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        //-- Draw matches
        Mat img_matches;
        drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;
        for( size_t i = 0; i < good_matches.size(); i++ ) {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }

        Mat H = findHomography( obj, scene, RANSAC );
        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f( (float)img_object.cols, 0 );
        obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
        obj_corners[3] = Point2f( 0, (float)img_object.rows );

        std::vector<Point2f> scene_corners(4);

        perspectiveTransform( obj_corners, scene_corners, H);


        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
            scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
            scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
            scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
            scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );


        // Resizes the img_match array to fit better on sceen
        resize(img_matches, resized_down, Size(down_width, down_height), INTER_LINEAR);

        //-- Show detected matches
        imshow("Good Matches & Object detection", resized_down);
        
        // Reading in any key presses during operation 
        char c=(char)waitKey(25);

        // The esc key is 27. So if pressed, breaks the loop
        if(c==27)
            break;

    }

// Releases the video so it can be opened again, less errors
cap.release();

// Closes all frames
destroyAllWindows();

// Goodbye
return 0;
}