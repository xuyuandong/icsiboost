#ifndef BOOST_TRAINING_H_
#define BOOST_TRAINING_H_

namespace icsiboost{

typedef struct _array array_t;
typedef struct _string string_t;
typedef struct _vector vector_t;
typedef struct _weakclassifier weakclassifier_t;
typedef struct _boostemplate boostemplate_t;

//global not const
const double smoothing = 0.5;
const int output_weights = 0;

#define EXP(a) exp(a)
#define LOG(a) log(a)
#define SQRT(number) sqrt(number)

#define y_l(x,y) (x->classes[y] == 1?1.0:-1.0)    // y_l() from the paper

#define b(x,y) (x->classes[y] == 1?1:0)           // b() from the paper (binary class match)

weakclassifier_t* train_text_stump(double min_objective, boostemplate_t* boostemplate, vector_t* examples, double** sum_of_weights, int num_classes);


weakclassifier_t* train_abstaining_text_stump(double min_objective, boostemplate_t* boostemplate, vector_t* examples, double** sum_of_weights, int num_classes);


weakclassifier_t* train_known_continuous_stump(double min_objective, boostemplate_t* boostemplate, vector_t* examples_vector, int num_classes);


weakclassifier_t* train_continuous_stump(double min_objective, boostemplate_t* boostemplate, vector_t* examples_vector, int num_classes);


/* compute max error rate */
double compute_max_error(vector_t* examples, int num_classes);

/* compute error rate AND update weights
   in testing conditions (dev or test set) sum_of_weights is NULL, so just compute error rate
   => need to be parallelized
*/
double compute_test_error(vector_t* classifiers, vector_t* examples, int classifier_id, int num_classes);



const double fmeasure_beta = 1;

int example_score_comparator(const void* a, const void* b);

double compute_max_fmeasure(vector_t* working_examples, int class_of_interest, double* threshold, double *recall_output, double *precision_output);


double compute_classification_error(vector_t* classifiers, vector_t* examples, int classifier_id, double** sum_of_weights, int num_classes, int use_known_continuous_stump, int use_abstaining_text_stump);


vector_t* load_examples_multilabel(const char* filename, vector_t* templates, vector_t* classes, double* class_priors, int feature_count_cutoff, int in_test);

} // end namespace

#endif /* BOOST_TRAINING_H_ */
