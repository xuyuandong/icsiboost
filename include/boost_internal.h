#ifndef BOOST_INTERNAL_H_
#define BOOST_INTERNAL_H_

#include <math.h>
#include "common.h"
#include "debug.h"
#include "vector.h"
#include "string.h"
#include "hashtable.h"
#include "mapped.h"
#include "array.h"
#include "file.h"

namespace icsiboost{

#define FEATURE_TYPE_IGNORE 0
#define FEATURE_TYPE_CONTINUOUS 1
#define FEATURE_TYPE_TEXT 2
#define FEATURE_TYPE_SCORED_TEXT 3
#define FEATURE_TYPE_SET 4

#define CLASSIFIER_TYPE_THRESHOLD 1
#define CLASSIFIER_TYPE_TEXT 2
#define CLASSIFIER_TYPE_EXTERNAL 3

typedef struct _boostemplate { // a column definition
	int column;
	string_t* name;
	int type;
    int text_expert_type;
    int text_expert_length;
    int no_unk_ngrams;
    string_t* drop_regex;
    int feature_count_cutoff;
	hashtable_t* dictionary;       // dictionary for token based boostemplate
	vector_t* tokens;              // the actual tokens (if the feature is text; contains tokeninfo)
	vector_t* values;              // continuous values (if the feature is continuous)
	int32_t* ordered;             // ordered example ids according to value for continuous columns
	vector_t* classifiers;         // classifiers related to that boostemplate (in classification mode only)
} boostemplate_t;

typedef struct example { // an instance
	int32_t* classes;
	int num_classes;
	double* score;                 // example score by class, updated iteratively
	double* weight;                // example weight by class
} example_t;

typedef struct test_example { // a test instance (not very memory efficient) WARNING: has to be "compatible" with example_t up to score[]
	int32_t* classes;
	int num_classes;
	double* score;
	float* continuous_features;
	vector_t** discrete_features;
} test_example_t;

typedef struct tokeninfo { // store info about a word (or token)
	int32_t id;
	char* key;
	size_t count;
	vector_t* examples;
} tokeninfo_t;

typedef struct _weakclassifier { // a weak learner
	boostemplate_t* boostemplate;          // the corresponding boostemplate
	int type;                      // type in CLASSIFIER_TYPE_TEXT or CLASSIFIER_TYPE_THRESHOLD
	int32_t token;                     // token for TEXT classifier
	int column;                    // redundant with boostemplate
	double threshold;              // threshold for THRESHOLD classifier
	double alpha;                  // alpha from the paper
	double objective;              // Z() value from minimization
	double *c0;                    // weight by class, unknown
	double *c1;                    // weight by class, token absent or below threshold
	double *c2;                    // weight by class, token present or above threshold
	string_t* external_filename;          // in case the classifier is external, use decisions from a file
	double** external_train_decisions;    // - for training (format is one example per line, one score per class in the order of the names file)
	double** external_dev_decisions;      // - for development
	double** external_test_decisions;     // - for testing
} weakclassifier_t;

vector_t* load_model(vector_t* templates, vector_t* classes, char* filename, int num_iterations_to_load);

void save_model(vector_t* classifiers, vector_t* classes, char* filename, int pack_model, int optimal_iterations);

#define TEXT_EXPERT_NGRAM 1
#define TEXT_EXPERT_SGRAM 2
#define TEXT_EXPERT_FGRAM 3
// warning: text experts SGRAM and FGRAM do not output anything if the window is larger than the actual number of words
array_t* text_expert(boostemplate_t* boostemplate, string_t* text);

} // end namespace

#endif // BOOST_INTERNAL_H_
