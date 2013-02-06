#include "boost_classifier.h"
#include <ctime>
#include <cstdio>

using namespace icsiboost;

int main(){
    BoostClassifier classifier;
    
    classifier.LoadNames("adult.names");

    //

    classifier.Reset();
    //
    classifier.Training("adult.data", "adult.mine");

    classifier.Training("adult.data", "adult.mine");

    classifier.LoadModel("adult.mine");
    classifier.Classify("adult.test", stderr);

    fprintf(stderr, "ok t c\n");

    classifier.Training("adult.data", "adult.mine");

    classifier.Reset();
    classifier.LoadModel("adult.shyp");
    classifier.Classify("adult.test", stderr);


    fprintf(stderr, "Time used = %.3lf\n", (double)clock() / CLOCKS_PER_SEC);

    return 0;
}
