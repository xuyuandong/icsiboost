#include "boost_classifier.h"
#include "boost_internal.h"
#include "boost_training.h"

namespace icsiboost{

	BoostClassifier::BoostClassifier(int output_mode) {
		// init options
        options.output_posteriors = output_mode;

        examples = NULL;
        dev_examples = NULL;
        test_examples = NULL;

		templates = vector_new(16);
		classes = NULL;

		classifiers = NULL;
		sum_of_alpha = 0;
		decision_threshold = (options.output_posteriors) ? 0.5 : 0.0;
	}

	BoostClassifier::~BoostClassifier(){
		ReleaseClasses();
		ReleaseExamples();
		ReleaseTemplates();
		ReleaseClassifiers();
	}

	void BoostClassifier::Reset(){
		if (templates != NULL){
			for(int i=0; i<templates->length; ++i)
			{
				boostemplate_t* bootemp = (boostemplate_t*)vector_get(templates,i);
				if (bootemp->classifiers != NULL){
					vector_free(bootemp->classifiers);
					bootemp->classifiers = NULL;
				}
			}
		}
		ReleaseExamples();
		ReleaseClassifiers();
		sum_of_alpha = 0;
	}

    void BoostClassifier::SetOptions(BoostOptions& new_options){
        options = new_options;
		decision_threshold = (options.output_posteriors) ? 0.5 : 0.0;
    } 

	int BoostClassifier::LoadNames(const char* fname){
		string_t* names_filename = string_new(fname); // .names file
		mapped_t* input = mapped_load_readonly(names_filename->data);
		if (input == NULL) return -1;

		hashtable_t* templates_by_name=hashtable_new();
		string_t* line = NULL;
		int line_num = 0;
		while((line = mapped_readline(input)) != NULL) // should add some validity checking !!!
		{
			if(string_match(line,"^(\\|| *$)","n")) // skip comments and blank lines
			{
				string_free(line);
				continue;
			}
			if(classes != NULL) // this line contains a boostemplate definition
			{
				array_t* parts = string_split(line, "(^ +| *: *| *\\.$)", NULL);
				boostemplate_t* boostemplate = (boostemplate_t*)MALLOC(sizeof(boostemplate_t));
				boostemplate->column = line_num - 1;
				boostemplate->name = (string_t*)array_get(parts, 0);
				string_t* type = (string_t*)array_get(parts, 1);
				string_t* name_options = NULL;
				if(parts->length > 2) {
					name_options = (string_t*)array_get(parts, 2);
				}
				boostemplate->dictionary = hashtable_new();
				boostemplate->tokens = vector_new(16);
				boostemplate->values = vector_new_float(16);
				boostemplate->classifiers = NULL;
				boostemplate->text_expert_type = TEXT_EXPERT_NGRAM;
				boostemplate->text_expert_length = 1;
				boostemplate->drop_regex = NULL;
				boostemplate->no_unk_ngrams = options.no_unk_ngrams;
				boostemplate->feature_count_cutoff = options.feature_count_cutoff;
				if(name_options != NULL) {
					array_t* option_list = string_split(name_options, "[ \t]+", NULL);
					int option_num;
					for(option_num = 0; option_num < option_list->length; option_num ++) {
						string_t* option = (string_t*) array_get(option_list, option_num);
						array_t* option_parts = string_split(option, "=", NULL);
						string_t* option_name = (string_t*) array_get(option_parts, 0);
						string_t* option_value = (string_t*) array_get(option_parts, 1);
						if (options.verbose)
							fprintf(stderr, "OPTION(%s): %s %s\n", boostemplate->name->data, option_name->data, option_value->data);
						if(string_eq_cstr(option_name, "expert_type")) {
							if(string_eq_cstr(option_value,"sgram"))
								boostemplate->text_expert_type=TEXT_EXPERT_SGRAM;
							else if(string_eq_cstr(option_value,"fgram"))
								boostemplate->text_expert_type=TEXT_EXPERT_FGRAM;
							else if(string_eq_cstr(option_value,"ngram"))
								boostemplate->text_expert_type=TEXT_EXPERT_NGRAM;
							else 
								die("unknown expert type \"%s\", line %d in %s", option->data, line_num+1, names_filename->data);
						} else if(string_eq_cstr(option_name, "drop")) {
							boostemplate->drop_regex = string_copy(option_value);
							if (options.verbose) 
								fprintf(stderr, "%s\n", boostemplate->drop_regex->data);
						} else if(string_eq_cstr(option_name, "no_unk")) {
							boostemplate->no_unk_ngrams=string_to_int32(option_value);
							if(boostemplate->no_unk_ngrams != 0 && boostemplate->no_unk_ngrams != 1)
								die("invalid value for no_unk \"%s\", line %d in %s", option->data, line_num+1, names_filename->data);
						} else if(string_eq_cstr(option_name, "expert_length")) {
							boostemplate->text_expert_length=string_to_int32(option_value);
							if(!boostemplate->text_expert_length>0)
								die("invalid expert length \"%s\", line %d in %s", option->data, line_num+1, names_filename->data);
						} else if(string_eq_cstr(option_name, "cutoff")) {
							boostemplate->feature_count_cutoff=string_to_int32(option_value);
							if (boostemplate->feature_count_cutoff<=0) 
								die("invalid cutoff \"%s\", line %d in %s", option->data, line_num+1, names_filename->data);
						} else {
							die("unknown option \"%s\", line %d in %s", option->data, line_num+1, names_filename->data);
						}
						string_array_free(option_parts);
					}
					string_array_free(option_list);
					string_free(name_options);
				}
				boostemplate->ordered=NULL;
				tokeninfo_t* unknown_token=(tokeninfo_t*)MALLOC(sizeof(tokeninfo_t));
				unknown_token->id=0;
				unknown_token->key=strdup("?");
				unknown_token->count=0;
				unknown_token->examples=NULL;
				vector_push(boostemplate->tokens,unknown_token);

				if(!strcmp(type->data, "continuous")) 
					boostemplate->type = FEATURE_TYPE_CONTINUOUS;
				else if(!strcmp(type->data, "text")) 
					boostemplate->type = FEATURE_TYPE_TEXT;
				else if(!strcmp(type->data, "scored text")) {
					boostemplate->type = FEATURE_TYPE_IGNORE;
					warn("ignoring column \"%s\" of type \"%s\", line %d in %s", boostemplate->name->data, type->data, line_num+1, names_filename->data);
				}
				else if(!strcmp(type->data, "ignore")) 
					boostemplate->type = FEATURE_TYPE_IGNORE;
				else 
					boostemplate->type = FEATURE_TYPE_SET;

				if(boostemplate->type == FEATURE_TYPE_SET)
				{
					array_t* values = string_split(type,"(^ +| *, *| *\\.$)", NULL);
					if(values->length <= 1)
                        die("invalid column definition \"%s\", line %d in %s", line->data, line_num+1, names_filename->data);
					for(int i=0; i<values->length; i++)
					{
						string_t* value=(string_t*)array_get(values,i);
						tokeninfo_t* tokeninfo=(tokeninfo_t*)MALLOC(sizeof(tokeninfo_t));
						tokeninfo->id=i+1; // skip unknown value (?)
						tokeninfo->key=strdup(value->data);
						tokeninfo->count=0;
						tokeninfo->examples=vector_new_int32_t(16);
						hashtable_set(boostemplate->dictionary, value->data, value->length, tokeninfo);
						vector_push(boostemplate->tokens,tokeninfo);
					}
					string_array_free(values);
				}

				if(hashtable_exists(templates_by_name, boostemplate->name->data, boostemplate->name->length)!=NULL)
					die("duplicate feature name \"%s\", line %d in %s",boostemplate->name->data, line_num+1, names_filename->data);
				vector_push(templates, boostemplate);
				hashtable_set(templates_by_name, boostemplate->name->data, boostemplate->name->length, boostemplate);
				string_free(type);
				array_free(parts);
				if(options.verbose)
					fprintf(stderr,"TEMPLATE: %d %s %d\n",boostemplate->column,boostemplate->name->data,boostemplate->type);
			}
			else // first line contains the classtype definitions
			{
				array_t* parts = string_split(line, "(^ +| *, *| *\\.$)", NULL);
				if(parts->length <= 1)
					die("invalid classes definition \"%s\", line %d in %s", line->data, line_num+1, names_filename->data);
				classes = vector_from_array(parts);
				array_free(parts);
				if(options.verbose)
				{
					fprintf(stderr,"CLASSES:");
					for (int i = 0; i < classes->length; i++)
						fprintf(stderr," %s",((string_t*)vector_get(classes,i))->data);
					fprintf(stderr,"\n");
				}
			}
			string_free(line);
			line_num++;
		}

		vector_optimize(templates);
		mapped_free(input);
		hashtable_free(templates_by_name);
		string_free(names_filename);
		return 1;
	}

	int BoostClassifier::LoadModel(const char* fname){
		string_t* model_name = string_new(fname); //.shyp file
		classifiers = load_model(templates,classes,model_name->data, options.test_time_iterations);
		if (classifiers == NULL) return -1;

		if(options.test_time_iterations > classifiers->length || options.test_time_iterations == -1) 
			options.test_time_iterations = classifiers->length;

		for(int i=0; i<classifiers->length; i++)
		{
			weakclassifier_t* classifier = (weakclassifier_t*)vector_get(classifiers,i);
			sum_of_alpha+=classifier->alpha;
		}
		string_free(model_name);
		return 1;
	}

	int BoostClassifier::Classify(std::string& feature, std::pair<int, float>& result){
		string_t* line = string_new(feature.c_str());

		array_t* array_of_tokens = string_split(line, " *, *", NULL);
		if(array_of_tokens->length < templates->length || array_of_tokens->length > templates->length + 1){
			fprintf(stderr, "wrong number of columns (%zd), \"%s\"", array_of_tokens->length, line->data);
			string_array_free(array_of_tokens);
			string_free(line);
			return -1;
		}

		double score[classes->length];
		CalcuFeatures(score, array_of_tokens);

		double max = 0;
		int argmax = 0;
		for(int l=0; l<classes->length; l++) {
			if(l == 0 || score[l] > max) {
				max = score[l];
				argmax = l;
			}
		}
		result.first = argmax;
		result.second = (float)max;

		string_array_free(array_of_tokens);
		string_free(line);

		return 1;
	}

	int BoostClassifier::Classify(const char* fname, FILE* fout){
		FILE* fin = fopen(fname, "r");
		if (fin == NULL){
			fprintf(stderr, "failed to open %s", fname);
			return -1;
		}
		double score[classes->length]; //scores for each class

		int line_num = 0;
		int num_examples = 0;
		string_t* line = NULL;
		while((line = string_readline(fin)) != NULL)
		{
			line_num++;
			string_chomp(line);
			if(string_match(line,"^(\\|| *$)","n")) // skip comments and blank lines
			{
				string_free(line);
				continue;
			}

			// get token array (feature)
			array_t* array_of_tokens = string_split(line, " *, *", NULL);
			if(array_of_tokens->length<templates->length || array_of_tokens->length>templates->length+1){
				fprintf(stderr, "wrong number of columns (%zd), \"%s\", line %d in %s", array_of_tokens->length, line->data, line_num, fname);
				string_array_free(array_of_tokens);
				string_free(line);
				continue;
			}

			// calculate each feature (token)
			CalcuFeatures(score, array_of_tokens);

			OutputResult(score, fout);

			string_array_free(array_of_tokens);
			string_free(line);
			num_examples++;
		}

		fclose(fin);
		return 1;
	}

	int BoostClassifier::Training(const char* train_fname, const char* model_fname, const char* test_fname, const char* dev_fname){
		// Load Data ==============================================================================================
		//vector_t *examples, *test_examples, *dev_examples;
		if (LoadSamples(train_fname, test_fname, dev_fname) < 0){
			fprintf(stderr, "failed to load training samples\n");
			return -1;
		}
        fprintf(stderr, "ok LoadSamples\n");
		// init =======================================================================================================
		// sum of weights by classes (to infer the other side of the partition in binary classifiers)
		double **sum_of_weights = (double**) MALLOC(sizeof(double*) * 2);
		sum_of_weights[0] = (double*) MALLOC(sizeof(double) * classes->length);
		sum_of_weights[1] = (double*) MALLOC(sizeof(double) * classes->length);
		for (int i = 0; i < classes->length; i++) {
			sum_of_weights[0][i] = 0.0;
			sum_of_weights[1][i] = 0.0;
		}
		for (int i = 0; i < examples->length; i++) {
			example_t* example = (example_t*) vector_get(examples, i);
			for (int l = 0; l < classes->length; l++) {
				sum_of_weights[b(example, l)][l] += example->weight[l];
			}
		}
		// multiple label vs. maxclass
		/*if (has_multiple_labels_per_example) {
			display_maxclass_error = 0;
		} else {
			display_maxclass_error = 1;
		}*/
		// training parameters
		int iteration = 0;
		double theoretical_error = 1.0;

		// create model file
		string_t* model_name = string_new(model_fname);

		// resume trained previous result ===========================================================================
		vector_t* tclassifiers = NULL;
		if (options.resume_training == 0) {
			tclassifiers = vector_new(options.maximum_iterations);
		} else {
			iteration = ResumeModel(model_name, &tclassifiers, sum_of_weights);
			if (iteration < 0) {
				fprintf(stderr, "failed to resume training with previous model.\n");
				fprintf(stderr, "start training from beginning.\n");
				iteration = 0;
			} else {
				iteration = tclassifiers->length;
			}
		}
		
        // main training process =================================================================
		for (; iteration < options.maximum_iterations; iteration++) {
			// train a decision and update classifiers
			TrainImpl(tclassifiers, iteration, theoretical_error, sum_of_weights);

			if (options.save_model_at_each_iteration)
				save_model(tclassifiers, classes, model_name->data, 0, 0);
		}

		// save model ====================================================================================
		save_model(tclassifiers, classes, model_name->data, 0, 0);
		string_free(model_name);

		// release memory ================================================================================
		free(sum_of_weights[0]);
		free(sum_of_weights[1]);
		free(sum_of_weights);

		ReleaseExamples();

		// release temporal classifiers
		for(int i=0; i<tclassifiers->length; i++)
		{
			weakclassifier_t* classifier = (weakclassifier_t*)vector_get(tclassifiers, i);
			if(classifier != NULL)
			{
				FREE(classifier->c0);
				FREE(classifier->c1);
				FREE(classifier->c2);
				FREE(classifier);
			}
		}
		vector_free(tclassifiers);

		return 1;
	}

	void BoostClassifier::CalcuFeatures(double* score, array_t* array_of_tokens){
		for(int l=0; l<classes->length; l++) 
			score[l]=0.0;

		for(int i=0; i<templates->length; i++)
		{
			boostemplate_t* boostemplate = (boostemplate_t*)vector_get(templates, i);
			string_t* token = (string_t*)array_get(array_of_tokens, i);

			if(boostemplate->type == FEATURE_TYPE_TEXT || boostemplate->type == FEATURE_TYPE_SET)
			{
				CalcuTextSetFeature(score, token, boostemplate);
			}
			else if(boostemplate->type == FEATURE_TYPE_CONTINUOUS)
			{
				CalcuContinuousFeature(score, token, boostemplate);
			}
			// FEATURE_TYPE_IGNORE
		}

		// normalize final score
		for(int l=0; l<classes->length; l++)
		{
			score[l]/=sum_of_alpha;
			if(options.output_posteriors)
			{
				score[l] = 1.0/(1.0+exp(-2*sum_of_alpha*score[l]));
				score[l] -= decision_threshold;
			}
		}
	}

	void BoostClassifier::CalcuTextSetFeature(double* score, string_t* token, boostemplate_t* boostemplate){
		hashtable_t* subtokens=hashtable_new();
		if(string_cmp_cstr(token,"?")!=0)
		{
			array_t* experts = (array_t*)text_expert(boostemplate, token);
			if(boostemplate->type == FEATURE_TYPE_SET && experts->length != 1) {
				die("value \"%s\" was not described in the .names file, (%s)", token->data, boostemplate->name->data);
			}
			for(int j=0; j<experts->length; j++)
			{
				string_t* expert = (string_t*)array_get(experts, j);
				tokeninfo_t* tokeninfo = (tokeninfo_t*)hashtable_get(boostemplate->dictionary, expert->data, expert->length);
				int id = -1;
				if(tokeninfo != NULL) {
					hashtable_set(subtokens, &tokeninfo->id, sizeof(tokeninfo->id), tokeninfo);
					id = tokeninfo->id;
				} else if(boostemplate->type == FEATURE_TYPE_SET) {
					die("token \"%s\" was not defined in names file", expert->data);
				}
				/*if(options.verbose) 
				fprintf(stderr, "  EXPERT(%s): \"%s\" (id=%d)\n", boostemplate->name->data, expert->data, id);*/
			}
			string_array_free(experts);
		}
		for(int j=0; j<boostemplate->classifiers->length; j++)
		{
			weakclassifier_t* classifier = (weakclassifier_t*)vector_get(boostemplate->classifiers, j);
			if(hashtable_get(subtokens, &classifier->token, sizeof(classifier->token))==NULL) {
				/*if(options.verbose) {
				fprintf(stderr, "    WEAKC %s=%d: C1 (absent)", classifier->boostemplate->name->data, classifier->token);
				for(int l=0; l<classes->length; l++) 
				fprintf(stderr, " %g", classifier->alpha*classifier->c1[l]);
				fprintf(stderr, "\n");
				}*/
				for(int l=0; l<classes->length; l++) 
					score[l]+=classifier->alpha*classifier->c1[l];
			} else {
				/*if(options.verbose) {
				fprintf(stderr, "    WEAKC %s=%d: C2 (present)", classifier->boostemplate->name->data, classifier->token);
				for(int l=0; l<classes->length; l++) 
				fprintf(stderr, " %g", classifier->alpha*classifier->c2[l]);
				fprintf(stderr, "\n");
				}*/
				for(int l=0; l<classes->length; l++) 
					score[l]+=classifier->alpha*classifier->c2[l];
			}
		}
		hashtable_free(subtokens);
	}

	void BoostClassifier::CalcuContinuousFeature(double* score, string_t* token, boostemplate_t* boostemplate){
		float value = NAN;
		if(string_cmp_cstr(token,"?") != 0)
			value=string_to_float(token);
		for(int j=0; j<boostemplate->classifiers->length; j++)
		{
			weakclassifier_t* classifier = (weakclassifier_t*)vector_get(boostemplate->classifiers, j);
			if(isnan(value)) {
				/*if(options.verbose) {
                 *
				fprintf(stderr, "    WEAKC %s=nan: C0 (unk)", classifier->boostemplate->name->data);
				for(int l=0; l<classes->length; l++) 
				fprintf(stderr, " %g", classifier->alpha*classifier->c1[l]);
				fprintf(stderr, "\n");
				}*/
				for(int l=0; l<classes->length; l++) 
					score[l]+=classifier->alpha*classifier->c0[l];
			} else if(value < classifier->threshold) {
				/*if(options.verbose) {
				fprintf(stderr, "    WEAKC %s<%f: C1 (<)", classifier->boostemplate->name->data, classifier->threshold);
				for(int l=0; l<classes->length; l++) 
				fprintf(stderr, " %g", classifier->alpha*classifier->c1[l]);
				fprintf(stderr, "\n");
				}*/
				for(int l=0; l<classes->length; l++) 
					score[l]+=classifier->alpha*classifier->c1[l];
			} else {
				/*if(options.verbose) {
				fprintf(stderr, "    WEAKC %s>=%f: C2 (>=)", classifier->boostemplate->name->data, classifier->threshold);
				for(int l=0; l<classes->length; l++) {
				fprintf(stderr, " %g", classifier->alpha*classifier->c2[l]);
				}
				fprintf(stderr, "\n");
				}*/
				for(int l=0; l<classes->length; l++) 
					score[l]+=classifier->alpha*classifier->c2[l];
			}
		}
	}

	void BoostClassifier::PrintResult(double* score){
		double max = 0.0;
		int max_class = -1000;
		for(int l=0; l<classes->length; l++)
		{
			fprintf(stderr,"% 5f", score[l] + decision_threshold);
			if (l < classes->length-1)
				fprintf(stderr," ");
			if (score[l]>max) {
				max = score[l];
				max_class = l;
			}
		}
		fprintf(stderr, "\nmaxclass: %d with score: % 5f\n", max_class, max);
	}

	void BoostClassifier::OutputResult(double* score, FILE* fp){
		if (fp == NULL)
			return;
		fprintf(fp,"\n");

		double max = 0;
		int argmax = -1000;
		for(int l=0; l<classes->length; l++) {
			if(l == 0 || score[l] > max) {
				max = score[l];
				argmax = l;
			}
		}

		fprintf(fp, "maxclass %d: % 4f\n", argmax, max);
		for(int l=0; l<classes->length; l++) {
			string_t* classtype = (string_t*)vector_get(classes, l);
			fprintf(fp,"%s(% 4f)    ", classtype->data, score[l] + decision_threshold);
		}

        fprintf(fp, "\n");
	}

	int BoostClassifier::LoadSamples(const char* train_fname, const char* test_fname, const char* dev_fname){
        if (train_fname == NULL)
            return -1;

		// training data
		double class_priors[classes->length];
		string_t* data_filename = string_new(train_fname);
		examples = load_examples_multilabel(data_filename->data, templates, classes, class_priors, options.feature_count_cutoff, 0);
		if (examples == NULL || examples->length == 0) {
			fprintf(stderr, "no training examples found in \"%s\"",	data_filename->data);
			return -1;
		}
		string_free(data_filename);

		// deactivated dev that don't work with the new indexed features (need separate handling)
        if (dev_fname != NULL) {
            string_t* dev_filename = string_new(dev_fname);
            dev_examples = load_examples_multilabel(dev_filename->data, templates, classes, NULL, 0, 1);
            if (dev_examples != NULL && dev_examples->length == 0) {
                warn("no dev examples found in \"%s\"", dev_filename->data);
                vector_free(dev_examples);
                dev_examples = NULL;
            }
            string_free(dev_filename);
        }

		// testing data while training
        if (test_fname != NULL) {
            string_t* test_filename = string_new(test_fname);
            test_examples = load_examples_multilabel(test_filename->data, templates, classes, NULL, 0, 1);
            if (test_examples != NULL && test_examples->length == 0) {
                warn("no test examples found in \"%s\"", test_filename->data);
                vector_free(test_examples);
                test_examples = NULL;
            }
            string_free(test_filename);
        }

		return 1;
	}

	int BoostClassifier::ResumeModel(string_t* model_name, vector_t** reclassifier, double** sum_of_weights){
		vector_t* classifiers = load_model(templates, classes, model_name->data, -1);
		if (classifiers == NULL)
			return -1;

		int iteration = 0;
		for (; iteration < classifiers->length; iteration++) {
			weakclassifier_t* classifier = (weakclassifier_t*)vector_get(classifiers, iteration);

			//train error and update weights
			double error = compute_classification_error(classifiers, examples, iteration, sum_of_weights, classes->length, options.use_known_continuous_stump, options.use_abstaining_text_stump);
			if (options.use_max_fmeasure)
				error = compute_max_fmeasure(examples, options.fmeasure_class_id, NULL,	NULL, NULL);
			if (options.display_maxclass_error)
				error = compute_max_error(examples, classes->length);

			double threshold = NAN;
			// dev error
			double dev_error = NAN;
			if (dev_examples != NULL) {
				// compute error rate on dev
				dev_error = compute_test_error(classifiers, dev_examples, iteration, classes->length);
				if (options.use_max_fmeasure)
					dev_error = compute_max_fmeasure(dev_examples, options.fmeasure_class_id, &threshold, NULL, NULL);
				if (options.display_maxclass_error)
					dev_error = compute_max_error(dev_examples, classes->length);
			}

			// test error
			double test_error = NAN;
			if (test_examples != NULL) {
				// compute error rate on test
				test_error = compute_test_error(classifiers, test_examples, iteration, classes->length);
				if (options.display_maxclass_error)
					test_error = compute_max_error(test_examples, classes->length);
				if (options.use_max_fmeasure)
					test_error = compute_max_fmeasure(test_examples, options.fmeasure_class_id, &threshold, NULL, NULL);
			}

			// display result
			fprintf(stderr, "rnd %d: wh-err= %f th-err= %f dev= %f test= %f train= %f\n",
				iteration + 1, classifier->objective, NAN, dev_error, test_error, error);
		} // end for iterations

		*reclassifier = classifiers;
		return iteration;
	}

	int BoostClassifier::TrainImpl(vector_t* classifiers, int iteration, double therr, double** sum_of_weights){
		double min_objective = 1.0;
		weakclassifier_t* classifier = NULL;
		for (int i = 0; i < templates->length; i++) { // find the best classifier
			boostemplate_t* boostemplate = (boostemplate_t*) vector_get(templates, i);
			weakclassifier_t* current = NULL;

			if (boostemplate->type == FEATURE_TYPE_CONTINUOUS) {
                //fprintf(stderr, "begin continous %d %d\n", i, options.use_known_continuous_stump);
				if (options.use_known_continuous_stump)
					current = train_known_continuous_stump(1.0, boostemplate, examples, classes->length);
				else
					current = train_continuous_stump(1.0, boostemplate, examples, classes->length);
                //fprintf(stderr, "end continous %d\n", i);
			}
			else if (boostemplate->type == FEATURE_TYPE_TEXT || boostemplate->type == FEATURE_TYPE_SET) {
                //fprintf(stderr, "begin text %d %d\n", i, options.use_abstaining_text_stump);
				if (options.use_abstaining_text_stump)
					current = train_abstaining_text_stump(1.0, boostemplate, examples, sum_of_weights, classes->length);
				else
					current = train_text_stump(1.0, boostemplate, examples,	sum_of_weights, classes->length);
                //fprintf(stderr, "end text %d\n", i);
			}

			// else => FEATURE_TYPE_IGNORE
			if (current == NULL) continue;

			if (current->objective - min_objective < -1e-11) {
				min_objective = current->objective;
				if (classifier != NULL) { // free previous classifier
					free(classifier->c0);
					free(classifier->c1);
					free(classifier->c2);
					free(classifier);
				}
				classifier = current;
			} else {
				free(current->c0);
				free(current->c1);
				free(current->c2);
				free(current);
			}
		} // end for each feature dimension (template->length)

		// record classifier
		vector_push(classifiers, classifier);

		// compute training error rate and update weights
        //fprintf(stderr, "begin compute error\n");
		double error = compute_classification_error(classifiers, examples,	iteration, sum_of_weights, classes->length, options.use_known_continuous_stump, options.use_abstaining_text_stump);
		double train_recall = NAN;
		double train_precision = NAN;
		if (options.use_max_fmeasure)
			error = compute_max_fmeasure(examples, options.fmeasure_class_id, NULL,	&train_recall, &train_precision);
		if (options.display_maxclass_error)
			error = compute_max_error(examples, classes->length);

		// dev error
        //fprintf(stderr, "begin dev error\n");
		double dev_recall = NAN;
		double dev_precision = NAN;
		double dev_error = NAN;
		double threshold = NAN;
		if (dev_examples != NULL) {
			dev_error = compute_test_error(classifiers, dev_examples, iteration, classes->length); // compute error rate on test
		if (options.use_max_fmeasure)
			dev_error = compute_max_fmeasure(dev_examples, options.fmeasure_class_id, &threshold, &dev_recall, &dev_precision);
		if (options.display_maxclass_error)
			dev_error = compute_max_error(dev_examples, classes->length);
		}
        
		// testing error
        //fprintf(stderr, "begin test error\n");
		double test_error = NAN;
		double test_recall = NAN;
		double test_precision = NAN;
		double test_threshold = NAN;
		if (test_examples != NULL) {
			// compute error rate on test
			test_error = compute_test_error(classifiers, test_examples, iteration,	classes->length);
			if (options.display_maxclass_error)
				test_error = compute_max_error(test_examples, classes->length);
			if (options.use_max_fmeasure)
				test_error = compute_max_fmeasure(test_examples, options.fmeasure_class_id,	&test_threshold, &test_recall, &test_precision);
		}

		// display result
        //fprintf(stderr, "begin display error\n");
		therr *= classifier->objective;
		if (options.use_max_fmeasure) {
			fprintf(stderr, "rnd %d: wh-err= %f th-err= %f dev= %f (R=%.3f, P=%.3f) test= %f (R=%.3f, P=%.3f) train= %f (R=%.3f, P=%.3f)\n",
				iteration + 1, classifier->objective, therr, dev_error, dev_recall, dev_precision, test_error, test_recall,
				test_precision, error, train_recall, train_precision);
		} else {
			fprintf(stderr, "rnd %d: wh-err= %f th-err= %f dev= %f test= %f train= %f\n",
				iteration + 1, classifier->objective, therr, dev_error, test_error, error);
		}

		fflush(stderr);

		return 1;
	}

	void BoostClassifier::ReleaseClassifiers(){
		if (classifiers != NULL){
			for (int i = 0; i < classifiers->length; ++i){
				weakclassifier_t* classifier = (weakclassifier_t*)vector_get(classifiers, i);
				if (classifier != NULL){
					free(classifier->c0);
					free(classifier->c1);
					free(classifier->c2);
					free(classifier);
				}
			}
			vector_free(classifiers);
			classifiers = NULL;
		}
	}

	void BoostClassifier::ReleaseClasses(){	// release classes
		if (classes != NULL){
			for (int i = 0; i < classes->length; ++i){
				string_free((string_t*)vector_get(classes, i));
			}
			vector_free(classes);
			classes = NULL;
		}
	}

	void BoostClassifier::ReleaseTemplates(){	// release templates
		if (templates != NULL){
			for(int i=0; i<templates->length; ++i)
			{
				boostemplate_t* bootemp = (boostemplate_t*)vector_get(templates,i);
				string_free(bootemp->name);
				hashtable_free(bootemp->dictionary);
				for(int j=0; j<bootemp->tokens->length; ++j)
				{
					tokeninfo_t* tokeninfo = (tokeninfo_t*) vector_get(bootemp->tokens, j);
					if(tokeninfo->examples != NULL)
						vector_free(tokeninfo->examples);
					free(tokeninfo->key);
					free(tokeninfo);
				}
				vector_free(bootemp->tokens);
				vector_free(bootemp->values);
				if(bootemp->ordered != NULL)
					free(bootemp->ordered);
				if(bootemp->classifiers != NULL)
					vector_free(bootemp->classifiers);
				free(bootemp);
			}
			vector_free(templates);
			templates = NULL;
		}
	}
	
	void BoostClassifier::ReleaseExamples() {
		if (dev_examples != NULL) {
			for (int i = 0; i < dev_examples->length; i++) {
				test_example_t* example = (test_example_t*)vector_get(dev_examples, i);
				free(example->continuous_features);
				for (int j = 0; j < templates->length; j++) {
					if (example->discrete_features[j] != NULL)
						vector_free(example->discrete_features[j]);
				}
				free(example->discrete_features);
				free(example->score);
				free(example);
			}
			vector_free(dev_examples);
            dev_examples = NULL;
		}
		if (test_examples != NULL) {
			for (int i = 0; i < test_examples->length; i++) {
				test_example_t* example = (test_example_t*)vector_get(test_examples, i);
				free(example->continuous_features);
				for (int j = 0; j < templates->length; j++) {
					if (example->discrete_features[j] != NULL)
						vector_free(example->discrete_features[j]);
				}free(example->discrete_features);
				free(example->score);
				free(example->classes);
				free(example);
			}
			vector_free(test_examples);
            test_examples = NULL;
		}
		if (examples != NULL) {
			for (int i = 0; i < examples->length; i++) {
				example_t* example = (example_t*) vector_get(examples, i);
				free(example->weight);
				free(example->score);
				free(example->classes);
				free(example);
			}
			vector_free(examples);
            examples = NULL;
		}
	}

} // end namespace
