#ifndef BOOST_CLASSIFIER_H_
#define BOOST_CLASSIFIER_H_

#include <string>

namespace icsiboost{

typedef struct _array array_t;
typedef struct _string string_t;
typedef struct _vector vector_t;
typedef struct _boostemplate boostemplate_t;

class BoostOptions{
public:
    // constructor
    BoostOptions(){
        output_posteriors = 0; // output score, not probability
		
        no_unk_ngrams = 0;
		feature_count_cutoff = 0;
		test_time_iterations = -1;

		resume_training = 0; // restart training, not resume previous one
        maximum_iterations = 10;
        use_abstaining_text_stump = 0;
        use_known_continuous_stump = 0;

        use_max_fmeasure = 0; // use error rate, not f-measure
        fmeasure_class_id = -1;

		display_maxclass_error = 0;
        save_model_at_each_iteration = 0; // not save model at each step, not support interrupt during training
		
        verbose = 0;
    }
public:
	// for test
    int output_posteriors;
    // for general
	int no_unk_ngrams;
	int feature_count_cutoff;
	int test_time_iterations;
    // for train
	int resume_training;
	int maximum_iterations;
    int use_abstaining_text_stump;
    int use_known_continuous_stump;
    // for optimize
	int use_max_fmeasure;	// f-measure
	int fmeasure_class_id; // if use_max_fmeasure==1, please set a class id
    // for display during training
	int display_maxclass_error;
    // for interruptible
	int save_model_at_each_iteration; //interruptible
    // whether or not print details information 
    // if it is set to 1, will display a lot of debug information
    int verbose;
};

class BoostClassifier{
public:
	BoostClassifier(int output_mode = 0);
	~BoostClassifier();

	// load feature names and classes names (each dimensional name)
	int LoadNames(const char* fname);

	// please call Reset() before each LoadModel()
	void Reset();
	// load model file ".shyp"
	int LoadModel(const char* fname);
	// classify and write result to output file
	int Classify(const char* fname, FILE* fout);
	// classify and return max class and its score as pair<int, float>
	int Classify(std::string& feature, std::pair<int, float>& result);

	// please SetOptions() before each Training(), if you don't want default settings
	// some options also related to classification, you can call before each Classify(),
	void SetOptions(BoostOptions& new_options);
	int Training(const char* train_fname, const char* model_fname, const char* test_fname = NULL, const char* dev_fname = NULL);

private:
	// for classification
	void CalcuFeatures(double* score, array_t* array_of_tokens);
	void CalcuTextSetFeature(double* score, string_t* token, boostemplate_t* boostemplate);
	void CalcuContinuousFeature(double* score, string_t* token, boostemplate_t* boostemplate);

	// for output classification result
	void PrintResult(double* score);
	void OutputResult(double* score, FILE* fp);

	// for training
	int LoadSamples(const char* train_fname, const char* test_fname, const char* dev_fname);
	int ResumeModel(string_t* model_name, vector_t** reclassifier, double** sum_of_weights);
	int TrainImpl(vector_t* classifiers, int iteration, double therr, double** sum_of_weights);

	// memory release
	void ReleaseClasses();
	void ReleaseExamples();
	void ReleaseTemplates();
	void ReleaseClassifiers();

	// options
    BoostOptions options;

	// samples for training
	vector_t* examples;
	vector_t* dev_examples;
	vector_t* test_examples;

	// structures
	vector_t* templates;
	vector_t* classes;

	// classifier
	vector_t* classifiers;
	double sum_of_alpha;
	double decision_threshold;
};

}

#endif
