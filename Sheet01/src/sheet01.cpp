#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

using namespace std;
using namespace cv;

/** classification header **/
#define NUM_ITERATIONS 1e4
#define STEP_SIZE 0.5
#define DELTA 1e-6

struct ClassificationParam{
        string posTrain, negTrain;
        string posTest, negTest;
};

// regression class for various regression methods
class LogisticRegression{
private:
        Mat train, test;                // each column is a feature vector
        Mat gtLabelTrain, gtLabelTest;  // row vector
        Mat phi;

        int loadFeatures(std::string& trainFile, std::string& testFile, Mat& feat, Mat& gtLabel);

public:
        LogisticRegression(ClassificationParam& param);
        int learnClassifier(); 
        int testClassifier();  
        float sigmoid(float a);
        ~LogisticRegression(){}
};

/** regression header **/
#define FIN_RBF_NUM_CLUST 300
#define RBF_SIGMA 1e-3

// reading input parameters
struct RegressionParam{
        std::string regressionTrain;
        std::string regressionTest;
};

// models for regression
class Model{
public:
        Mat phi;        // each row models wi
        Mat sigma_sq;   // column vector
        Mat codeBook;   // codebook for finite kernel reg.
};

// regression class for various regression methods
class Regression{
private:
        Mat   trainx, trainw;
        Mat   testx, testw;
        Model linear_reg, fin_rbf_reg, dual_reg;

        int loadFeatures(std::string& fileName, Mat& vecx, Mat& vecw);

public:
        Regression(RegressionParam& param);
        ~Regression(){}
        int trainLinearRegression(); 
        int trainFinite_RBF_KernelRegression(); 
        int trainDualRegression(); 

        int testLinearRegresssion(); 
        int testFinite_RBF_KernelRegression(); 
        int testDualRegression(); 

};

int main()
{
        RegressionParam rparam;
        rparam.regressionTrain = "./data/regression_train.txt";
        rparam.regressionTest  = "./data/regression_test.txt";

        Regression reg(rparam);

        // Linear regression
        reg.trainLinearRegression();
        reg.testLinearRegresssion();
        reg.trainFinite_RBF_KernelRegression();
        reg.testFinite_RBF_KernelRegression();
        reg.trainDualRegression();
        reg.testDualRegression();

        // Logistic regression
        ClassificationParam cparam;
        cparam.posTrain = "./data/bottle_train.txt";
        cparam.negTrain = "./data/horse_train.txt";
        cparam.posTest  = "./data/bottle_test.txt";
        cparam.negTest  = "./data/horse_test.txt";

        LogisticRegression cls(cparam);
        cls.learnClassifier();
        cls.testClassifier();

        return 0;
}


/** regression functions **/
Regression::Regression(RegressionParam& param){
        // load features
        loadFeatures(param.regressionTrain,trainx,trainw);
        loadFeatures(param.regressionTest,testx,testw);
                cout<<"features loaded successfully"<<endl;

        // model memory
        linear_reg.phi.create(trainx.rows,trainw.rows,CV_32F); 
        linear_reg.phi.setTo(0);
        linear_reg.sigma_sq.create(trainw.rows,1,CV_32F); 
        linear_reg.sigma_sq.setTo(0);
        
        fin_rbf_reg.phi.create(FIN_RBF_NUM_CLUST,trainw.rows,CV_32F);
        fin_rbf_reg.sigma_sq.create(trainw.rows,1,CV_32F);
        
        dual_reg.phi.create(trainx.cols,trainw.rows,CV_32F);
        dual_reg.sigma_sq.create(trainw.rows,1,CV_32F);

}
int Regression::loadFeatures(string& fileName, Mat& matx, Mat& matw){

        // init dimensions and file
        int numR, numC, dimW;
        ifstream iStream(fileName.c_str());
        if(!iStream){
                cout<<"cannot read feature file: "<<fileName<<endl;
                exit(-1);
        }

        // read file contents
        iStream >> numR;
        iStream >> numC;
        iStream >> dimW;
        matx.create(numC-dimW+1,numR,CV_32F); // each column is a feature
        matw.create(dimW,numR,CV_32F);        // each column is a vector to be regressed

        for(int r=0; r<numR; ++r){
                // read world data
                for(int c=0; c<dimW; ++c)
                iStream >> matw.at<float>(c,r);
                // read feature data
                matx.at<float>(0,r)=1; // re-adjust feature vector to accommodate the intercept
                for(int c=0; c<numC-dimW; ++c)
                iStream >> matx.at<float>(c+1,r);
        }
        iStream.close();

        return 0;
}

int Regression::trainLinearRegression()
{
        std::cout << determinant((trainx * trainx.t())) << std::endl; // determinant is zero!
        
        // Regressor for world variable w0
        Mat phi_cap_0 = (trainx * trainx.t()).inv(DECOMP_SVD) * trainx * (trainw.row(0).t());
        phi_cap_0.copyTo(this->linear_reg.phi.col(0));
        
        Mat term0 = trainw.row(0).t() - (trainx.t() * phi_cap_0);
        Mat sig_sq_cap_0 = (term0.t() * term0) / trainx.cols;
        sig_sq_cap_0.copyTo(this->linear_reg.sigma_sq.row(0));
        
        // Regressor for world variable w1
        Mat phi_cap_1 = (trainx * trainx.t()).inv(DECOMP_SVD) * trainx * (trainw.row(1).t());
        phi_cap_1.copyTo(this->linear_reg.phi.col(1));
        
        Mat term1 = trainw.row(1).t() - (trainx.t() * phi_cap_1);
        Mat sig_sq_cap_1 = (term1.t() * term1) / trainx.cols;
        sig_sq_cap_1.copyTo(this->linear_reg.sigma_sq.row(1));
        
        std::cout << "Training completed..." << std::endl;
}

int Regression::testLinearRegresssion()
{
        Mat preds = this->linear_reg.phi.t() * this->testx;
        std::cout << "preds shape: " << preds.size() << std::endl;
        Mat diff = preds - this->testw;
        Mat sq_diff = diff.mul(diff);
        std::cout << "sq_diff shape: " << sq_diff.size() << std::endl;
        
        Mat avg_sq_diff;
        reduce(sq_diff, avg_sq_diff, 1, CV_REDUCE_AVG);
        std::cout << "avg_sq_diff: " << avg_sq_diff << std::endl;
}

int Regression::trainFinite_RBF_KernelRegression()
{
        int codebook_dim = FIN_RBF_NUM_CLUST;
        float lambda = RBF_SIGMA;

        // use kmeans to find codebook
        Mat labels, centers;
        kmeans(trainx.t(), codebook_dim, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.001), 5, KMEANS_RANDOM_CENTERS, centers);
        
        Mat codebook = centers.t();
        this->fin_rbf_reg.codeBook = codebook;
        
        Mat Z;
        Z.create(codebook_dim, trainx.cols, trainx.type());
        for(int i = 0; i < trainx.cols; i++){ // for each training sample
                Mat xi = trainx.col(i);
                Mat_<float> zi(codebook_dim, 1, CV_32F);
                zi.setTo(0.0f);
                int z_idx = 0;
                for(int c = 0; c < codebook.cols; c++) { // for each codebook vector
                        Mat ci = codebook.col(c);
                        double sqrd_norm = norm(xi - ci, NORM_L2SQR);
                        zi(z_idx, 0) = exp(-(sqrd_norm / lambda));
                        z_idx++;
                }
                zi.copyTo(Z.col(i));
        }
        
        // now train in the usual way
        
        // Regressor for world variable w0
        Mat phi_cap_0 = (Z * Z.t()).inv(DECOMP_SVD) * Z * (trainw.row(0).t());
        phi_cap_0.copyTo(this->fin_rbf_reg.phi.col(0));

        Mat term0 = trainw.row(0).t() - (Z.t() * phi_cap_0);
        Mat sig_sq_cap_0 = (term0.t() * term0) / Z.cols;
        sig_sq_cap_0.copyTo(this->fin_rbf_reg.sigma_sq.row(0));
        
        // Regressor for world variable w1
        Mat phi_cap_1 = (Z * Z.t()).inv(DECOMP_SVD) * Z * (trainw.row(1).t());
        phi_cap_1.copyTo(this->fin_rbf_reg.phi.col(1));
        
        Mat term1 = trainw.row(1).t() - (Z.t() * phi_cap_1);
        Mat sig_sq_cap_1 = (term1.t() * term1) / Z.cols;
        sig_sq_cap_1.copyTo(this->fin_rbf_reg.sigma_sq.row(1));
        
        std::cout << "Training completed..." << std::endl;
}

int Regression::testFinite_RBF_KernelRegression()
{
        int codebook_dim = FIN_RBF_NUM_CLUST;
        float lambda = RBF_SIGMA;

        // convert the test data dimension to match the codebook's dimension
        Mat Z, codebook = this->fin_rbf_reg.codeBook;
        Z.create(codebook_dim, testx.cols, testx.type());
        for(int i = 0; i < testx.cols; i++){ // for each training sample
                Mat xi = testx.col(i);
                Mat_<float> zi(codebook_dim, 1, CV_32F);
                zi.setTo(0.0f);
                int z_idx = 0;
                for(int c = 0; c < codebook.cols; c++) { // for each codebook vector
                        Mat ci = codebook.col(c);
                        double sqrd_norm = norm(xi - ci, NORM_L2SQR);
                        zi(z_idx, 0) = exp(-(sqrd_norm / lambda));
                        z_idx++;
                }
                zi.copyTo(Z.col(i));
        }
        
        Mat preds = this->fin_rbf_reg.phi.t() * Z;
        std::cout << "preds shape: " << preds.size() << std::endl;
        Mat diff = preds - this->testw;
        Mat sq_diff = diff.mul(diff);
        std::cout << "sq_diff shape: " << sq_diff.size() << std::endl;
        
        Mat avg_sq_diff;
        reduce(sq_diff, avg_sq_diff, 1, CV_REDUCE_AVG);
        std::cout << "avg_sq_diff: " << avg_sq_diff << std::endl;
}

int Regression::trainDualRegression()
{
        // Regressor for world variable w0
        Mat shi_cap_0 = (trainx.t() * trainx).inv(DECOMP_SVD) * (trainw.row(0).t()); // dual
        shi_cap_0.copyTo(this->dual_reg.phi.col(0));
        
        Mat term0 = trainw.row(0).t() - (trainx.t() * trainx * shi_cap_0);
        Mat sig_sq_cap_0 = (term0.t() * term0) / trainx.cols;
        sig_sq_cap_0.copyTo(this->dual_reg.sigma_sq.row(0));
        
        // Regressor for world variable w1
        Mat shi_cap_1 = (trainx.t() * trainx).inv(DECOMP_SVD) * (trainw.row(1).t()); //  dual
        
        shi_cap_1.copyTo(this->dual_reg.phi.col(1));    
        
        Mat term1 = trainw.row(1).t() - (trainx.t() * trainx * shi_cap_1);
        Mat sig_sq_cap_1 = (term1.t() * term1) / trainx.cols;
        sig_sq_cap_1.copyTo(this->dual_reg.sigma_sq.row(1));
        
        std::cout << "Training completed..." << std::endl;
}

int Regression::testDualRegression()
{
        Mat phi_cap = trainx * this->dual_reg.phi;
        
        Mat preds = phi_cap.t() * this->testx;
//         std::cout << "preds shape: " << preds.size() << std::endl;
        Mat diff = preds - this->testw;
        Mat sq_diff = diff.mul(diff);
        std::cout << "sq_diff shape: " << sq_diff.size() << std::endl;
        
        Mat avg_sq_diff;
        reduce(sq_diff, avg_sq_diff, 1, CV_REDUCE_AVG);
        std::cout << "avg_sq_diff: " << avg_sq_diff << std::endl;
}


/** classification functions **/
LogisticRegression::LogisticRegression(ClassificationParam& param){

        loadFeatures(param.posTrain,param.negTrain,train,gtLabelTrain);
        loadFeatures(param.posTest,param.negTest,test,gtLabelTest);
}

int LogisticRegression::loadFeatures(string& trainPos, string& trainNeg, Mat& feat, Mat& gtL){

        ifstream iPos(trainPos.c_str());
        if(!iPos) {
                cout<<"error reading train file: "<<trainPos<<endl;
                exit(-1);
        }
        ifstream iNeg(trainNeg.c_str());
        if(!iNeg) {
                cout<<"error reading test file: "<<trainNeg<<endl;
                exit(-1);
        }

        int rPos, rNeg, cPos, cNeg;
        iPos >> rPos;
        iPos >> cPos;
        iNeg >> rNeg;
        iNeg  >> cNeg;

        if(cPos != cNeg){
                cout<<"Number of features in pos and neg classes unequal"<<endl;
                exit(-1);
        }
        feat.create(cPos+1,rPos+rNeg,CV_32F); // each column is a feat vect
        gtL.create(1,rPos+rNeg,CV_32F);       // row vector


        // load positive examples
        for(int idr=0; idr<rPos; ++idr){
                gtL.at<float>(0,idr) = 1;
                feat.at<float>(0,idr) = 1;
                for(int idc=0; idc<cPos; ++idc){
                iPos >> feat.at<float>(idc+1,idr);
                }
        }

        // load negative examples
        for(int idr=0; idr<rNeg; ++idr){
                gtL.at<float>(0,rPos+idr) = 0;
                feat.at<float>(0,rPos+idr) = 1;
                for(int idc=0; idc<cNeg; ++idc){
                iNeg >> feat.at<float>(idc+1,rPos+idr);
                }
        }

        iPos.close();
        iNeg.close();

        return 0;
}

float LogisticRegression::sigmoid(float a){
        return 1.0f/(1+exp(-a));
}


int LogisticRegression::learnClassifier()
{
        int D = this->train.rows; // the dimension D of the parameter vector
        
        std::cout << "data dimension: " << D << std::endl;

        // The parameter vector phi
        phi.create(D, 1, CV_32F);
        phi.setTo(1.0f); // set uniform value for all parameters
        
        Mat dL, ddL; // the derivative matrices
        dL.create(D, 1, CV_32F);
        ddL.create(D, D, CV_32F);
        
        float L = 0.0f; // the loss
        
        // perform optimization
        for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
                
                // clear accumulators
                dL.setTo(0.0f);
                ddL.setTo(0.0f);
                L = 0.0f;
                
                // for each training sample
                for(int i = 0; i < this->train.cols; i++){
                      Mat xi = this->train.col(i);
                      
                      Mat tmp = phi.t() * xi; // The dot product of the parameter and training sample vectors
                      float ai = tmp.at<float>(0,0); // get the sigmoid argument
                      float wi = this->gtLabelTrain.at<float>(0, i);
                      // add a small value DELTA in log to avoid NaN
                      float L_update = ((wi * log(this->sigmoid(ai) + DELTA)) + ((1 - wi) * log(1 - this->sigmoid(ai) + DELTA)));
                      L = L + L_update; // the log likelihood
                      dL = dL + (xi * (this->sigmoid(ai) - wi)); // The 1st derivative
//                       ddL = ddL + ( xi * xi.t() * (this->sigmoid(ai) * (1 - this->sigmoid(ai)))); // The 2nd derivative
                }
                dL = -1.0 * dL;
//                 ddL = -1.0 * ddL;
//                 Mat ddL_inv = ddL.inv(DECOMP_SVD); // invert the 2nd derivative; takes very long!
                // update parameter estimate based on current sample
                // Applying Newton's method
//                 phi = phi + STEP_SIZE * ddL_inv * dL;
                // Applying simple gradient ascent method
                phi = phi + STEP_SIZE * dL;
                if(iter % 100 == 0){
                        std::cout << iter << "\t";
                        std::cout << "Log Likelihood: " << L << std::endl;
                }
        }
        std::cout << std::endl;
        /*
                The number of iterations remaining fixed at 1e4, we tested the following step sizes: 0.01, 0.05, 0.1, 0.5;
                we got accuracy values 60%, 75%, 75% and 90% respectively. The final likelihood value was the highest with 
                step-size 0.5, with the likelihood value steadily decreasing with decreasing step-size. 
         */
        return 0;
}

int LogisticRegression::testClassifier()
{
        std::cout << "test data count: " << this->test.cols << std::endl;
        int tp = 0, fp = 0, tn = 0, fn = 0; // accuracy stats
        // for each test sample
        for(int i = 0; i < this->test.cols; i++){
                Mat xi = this->test.col(i); // the test sample
                Mat tmp = phi.t() * xi;
                float ai = tmp.at<float>(0,0);
                int pred = cvRound(this->sigmoid(ai)); // the prediction is in [0,1]. Round it off to 0 or 1 to compare it to test label. The decision boundary is 0.5
                int tru_labl = this->gtLabelTest.at<float>(0,i);
                if(pred == 1 && tru_labl == 1){
                        tp++;
                }else if(pred == 1 && tru_labl == 0) {
                        fp++;
                }else if(pred == 0 && tru_labl == 1) {
                        fn++;
                }else if(pred == 0 && tru_labl == 0) {
                        tn++;
                }
        }
        float accuracy = (1.0 * (tp + tn)) / (tp + tn + fp + fn);
        std::cout << "tp= " << tp << ", tn= " << tn << ", fp= " << fp << ", fn= " << fn << std::endl;
        std::cout << "accuracy= " << accuracy << std::endl;
        return 0;
}


