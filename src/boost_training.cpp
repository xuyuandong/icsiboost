
#include "boost_internal.h"
#include "boost_training.h"

#include <float.h>

namespace icsiboost{

//global not const
int enforce_anti_priors = 0;
int example_score_comparator_class = 0;
int has_multiple_labels_per_example = 0;


weakclassifier_t* train_text_stump(double min_objective, boostemplate_t* boostemplate, vector_t* examples, double** sum_of_weights, int num_classes)
{
	int i,l;
	int column=boostemplate->column;
	size_t num_tokens=boostemplate->tokens->length;
	int32_t t;

	double* weight[2][num_classes]; // weight[b][label][token] (=> D() in the paper), only for token presence (absence is inferred from sum_of_weights)
	for(l=0;l<num_classes;l++)
	{
		weight[0][l] = (double*)MALLOC(sizeof(double)*num_tokens);
		weight[1][l] = (double*)MALLOC(sizeof(double)*num_tokens);
	}
	for(t=1;t<num_tokens;t++)  // initialize
	{
		for(l=0;l<num_classes;l++)
		{
			weight[0][l][t]=0.0;
			weight[1][l][t]=0.0;
		}

		tokeninfo_t* tokeninfo=(tokeninfo_t*)vector_get(boostemplate->tokens, t);
		//fprintf(stderr, "%s [%s] %d\n",boostemplate->name->data, tokeninfo->key, tokeninfo->examples->length);

		for(i=0;i<tokeninfo->examples->length;i++) // compute the presence weights
		{
			example_t* example = (example_t*)vector_get(examples, vector_get_int32_t(tokeninfo->examples, i));
			for(l=0;l<num_classes;l++)
			{
				weight[b(example,l)][l][t]+=example->weight[l];
			}
		}
	}
	weakclassifier_t* classifier=NULL; // init an empty classifier
	classifier = (weakclassifier_t*)MALLOC(sizeof(weakclassifier_t));
	classifier->boostemplate=boostemplate;
	classifier->threshold=NAN;
	classifier->alpha=1.0;
	classifier->type=CLASSIFIER_TYPE_TEXT;
	classifier->token=0;
	classifier->column=column;
	classifier->objective=1.0;
	classifier->c0 = (double*)MALLOC(sizeof(double)*num_classes);
	classifier->c1 = (double*)MALLOC(sizeof(double)*num_classes);
	classifier->c2 = (double*)MALLOC(sizeof(double)*num_classes);
	double epsilon=smoothing/(num_classes*examples->length);

	//min_objective=1;
	for(t=1;t<num_tokens;t++)
	{
		double objective=0;
		for(l=0;l<num_classes;l++) // compute the objective function Z()=sum_j(sum_l(SQRT(W+*W-))
		{
			/*if(weight[0][l][t]<0)weight[0][l][t]=0.0;
			if(weight[1][l][t]<0)weight[1][l][t]=0.0;*/
			objective+=SQRT((sum_of_weights[1][l]-weight[1][l][t])*(sum_of_weights[0][l]-weight[0][l][t]));
			objective+=SQRT(weight[1][l][t]*weight[0][l][t]);
		}
		objective*=2;
		//fprintf(stderr,"DEBUG: column=%d token=%d obj=%f\n",column,t,objective);
		if(objective-min_objective<-1e-11) // select the argmin()
		{
			min_objective=objective;
			classifier->token=t;
			classifier->objective=objective;
			for(l=0;l<num_classes;l++)  // update c0, c1 and c2 => c0 and c1 are the same for text stumps
			{
				classifier->c0[l]=0.5*LOG((sum_of_weights[1][l]-weight[1][l][t]+epsilon)/(sum_of_weights[0][l]-weight[0][l][t]+epsilon));
				classifier->c1[l]=classifier->c0[l];
				//classifier->c0[l]=0;
				//classifier->c1[l]=0.5*LOG((sum_of_weights[1][l]-weight[1][l][t]+epsilon)/(sum_of_weights[0][l]-weight[0][l][t]+epsilon));
				classifier->c2[l]=0.5*LOG((weight[1][l][t]+epsilon)/(weight[0][l][t]+epsilon));
			}
		}
	}
	for(l=0;l<num_classes;l++) // free memory
	{
		FREE(weight[0][l]);
		FREE(weight[1][l]);
	}
	//tokeninfo_t* info=vector_get(boostemplate->tokens,classifier->token);
	//fprintf(stderr,"DEBUG: column=%d token=%s obj=%f %s\n",column,info->key,classifier->objective,boostemplate->name->data);
	if(classifier->token==0) // no better classifier has been found
	{
		FREE(classifier->c0);
		FREE(classifier->c1);
		FREE(classifier->c2);
		FREE(classifier);
		return NULL;
	}
	return classifier;
}

weakclassifier_t* train_abstaining_text_stump(double min_objective, boostemplate_t* boostemplate, vector_t* examples, double** sum_of_weights, int num_classes)
{
	int i,l;
	int column=boostemplate->column;
	size_t num_tokens=boostemplate->tokens->length;
	int32_t t;

	double* weight[2][num_classes]; // weight[b][label][token] (=> D() in the paper), only for token presence (absence is inferred from sum_of_weights)
	for(l=0;l<num_classes;l++)
	{
		weight[0][l] = (double*)MALLOC(sizeof(double)*num_tokens);
		weight[1][l] = (double*)MALLOC(sizeof(double)*num_tokens);
	}
	for(t=1;t<num_tokens;t++)  // initialize
	{
		for(l=0;l<num_classes;l++)
		{
			weight[0][l][t]=0.0;
			weight[1][l][t]=0.0;
		}
		tokeninfo_t* tokeninfo=(tokeninfo_t*)vector_get(boostemplate->tokens, t);
		//fprintf(stderr,"%s [%s] %d\n",boostemplate->name->data,tokeninfo->key,tokeninfo->examples->length);
		for(i=0;i<tokeninfo->examples->length;i++) // compute the presence weights
		{
			example_t* example=(example_t*)vector_get(examples,vector_get_int32_t(tokeninfo->examples,i));
			for(l=0;l<num_classes;l++)
			{
				weight[b(example,l)][l][t]+=example->weight[l];
			}
		}
	}
	weakclassifier_t* classifier=NULL; // init an empty classifier
	classifier = (weakclassifier_t*)MALLOC(sizeof(weakclassifier_t));
	classifier->boostemplate=boostemplate;
	classifier->threshold=NAN;
	classifier->alpha=1.0;
	classifier->type=CLASSIFIER_TYPE_TEXT;
	classifier->token=0;
	classifier->column=column;
	classifier->objective=1.0;
	classifier->c0 = (double*)MALLOC(sizeof(double)*num_classes);
	classifier->c1 = (double*)MALLOC(sizeof(double)*num_classes);
	classifier->c2 = (double*)MALLOC(sizeof(double)*num_classes);
	double epsilon=smoothing/(num_classes*examples->length);

	//min_objective=1;
	for(t=1;t<num_tokens;t++)
	{
		double objective=0;
		double w0=0;
		for(l=0;l<num_classes;l++) // compute the objective function Z()=sum_j(sum_l(SQRT(W+*W-))
		{
			objective+=SQRT(weight[1][l][t]*weight[0][l][t]);
			//objective+=SQRT((sum_of_weights[1][l]-weight[1][l][t])*(sum_of_weights[0][l]-weight[0][l][t]));
			w0+=sum_of_weights[0][l]-weight[0][l][t]+sum_of_weights[1][l]-weight[1][l][t];
		}

		objective*=2;
		objective+=w0;
		//fprintf(stderr,"DEBUG: column=%d token=%d obj=%f w0=%f\n",column,t,objective, w0);

		if(objective-min_objective<-1e-11) // select the argmin()
		{
			min_objective=objective;
			classifier->token=t;
			classifier->objective=objective;
			for(l=0;l<num_classes;l++)  // update c0, c1 and c2 => c0 and c1 are the same for text stumps
			{
				classifier->c0[l]=0.0;
				classifier->c1[l]=0.0;
				classifier->c2[l]=0.5*LOG((weight[1][l][t]+epsilon)/(weight[0][l][t]+epsilon));
			}
		}
	}
	for(l=0;l<num_classes;l++) // free memory
	{
		FREE(weight[0][l]);
		FREE(weight[1][l]);
	}
	//tokeninfo_t* info=vector_get(boostemplate->tokens,classifier->token);
	//fprintf(stderr,"DEBUG: column=%d token=%s obj=%f %s\n",column,info->key,classifier->objective,boostemplate->name->data);
	if(classifier->token==0) // no better classifier has been found
	{
		FREE(classifier->c0);
		FREE(classifier->c1);
		FREE(classifier->c2);
		FREE(classifier);
		return NULL;
	}
	return classifier;
}

// local comparator
struct KeyVal{
    int32_t key;
    float value;
};

int kv_comparator(const void* _a, const void* _b)
{
    KeyVal* kva = (KeyVal*) _a;
    KeyVal* kvb = (KeyVal*) _b;
    float aa_value = kva->value;
    float bb_value = kvb->value;
    if(isnan(aa_value) || aa_value>bb_value)
        return 1; // put the NAN (unknown values) at the end of the list
    if(isnan(bb_value) || aa_value<bb_value)
        return -1;
    return 0;
}

weakclassifier_t* train_known_continuous_stump(double min_objective, boostemplate_t* boostemplate, vector_t* examples_vector, int num_classes)
{
	size_t i,j,l;
	int column=boostemplate->column;
	float* values=(float*)boostemplate->values->data;
	int32_t* ordered=boostemplate->ordered;
	example_t** examples=(example_t**)examples_vector->data;
	if(ordered==NULL) // only order examples once, then keep the result in the boostemplate
	{
        KeyVal* kvarr = (KeyVal*)MALLOC(sizeof(KeyVal)*examples_vector->length);
		for(int32_t index = 0; index < examples_vector->length; index++){
            kvarr[index].key = index;
            kvarr[index].value = values[index];
        }
		qsort(kvarr, examples_vector->length, sizeof(KeyVal), kv_comparator);

		ordered = (int32_t*)MALLOC(sizeof(int32_t)*examples_vector->length);
        for (int32_t index = 0; index < examples_vector->length; index++)
            ordered[index] = kvarr[index].key;
        FREE(kvarr);

		boostemplate->ordered=ordered;
	}

	double weight[3][2][num_classes]; // D(j,b,l)
	for(j=0;j<3;j++)
		for(l=0;l<num_classes;l++)
		{
			weight[j][0][l]=0.0;
			weight[j][1][l]=0.0;
		}
	//double sum_of_unknowns = 0;
	for(i=0;i<examples_vector->length;i++) // compute the "unknown" weights and the weight of examples after threshold
	{
		int32_t example_id=ordered[i];
		example_t* example=examples[example_id];
		//fprintf(stderr,"%d %f\n",column,vector_get_float(example->features,column));
		for(l=0;l<num_classes;l++)
		{
			if(isnan(values[example_id])) {
				//sum_of_unknowns += example->weight[l];
				weight[0][b(example,l)][l]+=example->weight[l];
			} else
				weight[2][b(example,l)][l]+=example->weight[l];
		}
	}
	weakclassifier_t* classifier=NULL; // new classifier
	classifier = (weakclassifier_t*)MALLOC(sizeof(weakclassifier_t));
	classifier->boostemplate=boostemplate;
	classifier->threshold=NAN;
	classifier->alpha=1.0;
	classifier->type=CLASSIFIER_TYPE_THRESHOLD;
	classifier->token=0;
	classifier->column=column;
	classifier->objective=1.0;
	classifier->c0 = (double*)MALLOC(sizeof(double)*num_classes);
	classifier->c1 = (double*)MALLOC(sizeof(double)*num_classes);
	classifier->c2 = (double*)MALLOC(sizeof(double)*num_classes);
	double epsilon=smoothing/(num_classes*examples_vector->length);

	for(i=0;i<examples_vector->length-1;i++) // compute the objective function at every possible threshold (in between examples)
	{
		int32_t example_id=ordered[i];
		example_t* example=examples[example_id];
		//fprintf(stderr,"%zd %zd %f\n",i,vector_get_int32_t(ordered,i),vector_get_float(boostemplate->values,example_id));
		if(isnan(values[example_id]))
			break; // skip unknown values
		//example_t* next_example=(example_t*)vector_get(examples,(size_t)next_example_id);
		for(l=0;l<num_classes;l++) // update the objective function by putting the current example the other side of the threshold
		{
			weight[1][b(example,l)][l]+=example->weight[l];
			weight[2][b(example,l)][l]-=example->weight[l];
		}
		int next_example_id=ordered[i+1];
		if(values[example_id]==values[next_example_id])continue; // same value
		double objective=0;
		double w0 = 0.0;
		for(l=0;l<num_classes;l++) // compute objective Z()
		{
			//objective+=SQRT(weight[0][1][l]*weight[0][0][l]);
			objective+=SQRT(weight[1][1][l]*weight[1][0][l]);
			objective+=SQRT(weight[2][1][l]*weight[2][0][l]);
			w0 += weight[0][0][l]+weight[0][1][l];
		}
		objective*=2;
		objective+=w0;
		//fprintf(stderr,"DEBUG: column=%d threshold=%f obj=%f\n",column,(vector_get_float(next_example->features,column)+vector_get_float(example->features,column))/2,objective);
		if(objective-min_objective<-1e-11) // get argmin
		{
			classifier->objective=objective;
			classifier->threshold=((double)values[next_example_id]+(double)values[example_id])/2.0; // threshold between current and next example
			if(isnan(classifier->threshold))die("threshold is nan, column=%d, objective=%f, i=%zd",column,objective,i); // should not happend
			//fprintf(stderr," %d:%d:%f",column,i,classifier->threshold);
			min_objective=objective;
			for(l=0;l<num_classes;l++) // update class weight
			{
				classifier->c0[l]=0.0;
				classifier->c1[l]=0.5*LOG((weight[1][1][l]+epsilon)/(weight[1][0][l]+epsilon));
				classifier->c2[l]=0.5*LOG((weight[2][1][l]+epsilon)/(weight[2][0][l]+epsilon));
			}
		}
	}
	//fprintf(stderr,"DEBUG: column=%d threshold=%f obj=%f %s\n",column,classifier->threshold,classifier->objective,boostemplate->name->data);
	if(isnan(classifier->threshold)) // not found a better classifier
	{
		FREE(classifier->c0);
		FREE(classifier->c1);
		FREE(classifier->c2);
		FREE(classifier);
		return NULL;
	}
	return classifier;
}

weakclassifier_t* train_continuous_stump(double min_objective, boostemplate_t* boostemplate, vector_t* examples_vector, int num_classes)
{
	size_t i,j,l;
	int column=boostemplate->column;
	float* values=(float*)boostemplate->values->data;
	int32_t* ordered=boostemplate->ordered;
	example_t** examples=(example_t**)examples_vector->data;
	if(ordered==NULL) // only order examples once, then keep the result in the boostemplate
	{
        KeyVal* kvarr = (KeyVal*)MALLOC(sizeof(KeyVal)*examples_vector->length);
		for(int32_t index = 0; index < examples_vector->length; index++){
            kvarr[index].key = index;
            kvarr[index].value = values[index];
        }
		qsort(kvarr, examples_vector->length, sizeof(KeyVal), kv_comparator);

		ordered = (int32_t*)MALLOC(sizeof(int32_t)*examples_vector->length);
        for (int32_t index = 0; index < examples_vector->length; index++)
            ordered[index] = kvarr[index].key;
        FREE(kvarr);
        
        boostemplate->ordered=ordered;
	}

	double weight[3][2][num_classes]; // D(j,b,l)
	for(j=0;j<3;j++)
		for(l=0;l<num_classes;l++)
		{
			weight[j][0][l]=0.0;
			weight[j][1][l]=0.0;
		}
	for(i=0;i<examples_vector->length;i++) // compute the "unknown" weights and the weight of examples after threshold
	{
		int32_t example_id=ordered[i];
		example_t* example=examples[example_id];
		//fprintf(stderr,"%d %f\n",column,vector_get_float(example->features,column));
		for(l=0;l<num_classes;l++)
		{
			if(isnan(values[example_id]))
				weight[0][b(example,l)][l]+=example->weight[l];
			else
				weight[2][b(example,l)][l]+=example->weight[l];
		}
	}
	weakclassifier_t* classifier=NULL; // new classifier
	classifier = (weakclassifier_t*)MALLOC(sizeof(weakclassifier_t));
	classifier->boostemplate=boostemplate;
	classifier->threshold=NAN;
	classifier->alpha=1.0;
	classifier->type=CLASSIFIER_TYPE_THRESHOLD;
	classifier->token=0;
	classifier->column=column;
	classifier->objective=1.0;
	classifier->c0 = (double*)MALLOC(sizeof(double)*num_classes);
	classifier->c1 = (double*)MALLOC(sizeof(double)*num_classes);
	classifier->c2 = (double*)MALLOC(sizeof(double)*num_classes);
	double epsilon=smoothing/(num_classes*examples_vector->length);

    double objective=0; // compute objective for threshold below any known value
    for(l=0;l<num_classes;l++) // compute objective Z()
    {
        double w0 = weight[0][1][l]*weight[0][0][l]; if(w0 > 0) objective += SQRT(w0);
        double w1 = weight[1][1][l]*weight[1][0][l]; if(w1 > 0) objective += SQRT(w1);
        double w2 = weight[2][1][l]*weight[2][0][l]; if(w2 > 0) objective += SQRT(w2);
    }
    objective*=2;
    if(objective-min_objective<-1e-11) // get argmin
    {
        classifier->objective=objective;
        classifier->threshold=-DBL_MAX; // -infinity

        min_objective=objective;
        for(l=0;l<num_classes;l++) // update class weight
        {
            classifier->c0[l]=0.5*LOG((weight[0][1][l]+epsilon)/(weight[0][0][l]+epsilon));
            classifier->c1[l]=0.5*LOG((weight[1][1][l]+epsilon)/(weight[1][0][l]+epsilon));
            classifier->c2[l]=0.5*LOG((weight[2][1][l]+epsilon)/(weight[2][0][l]+epsilon));
        }
    }
	for(i=0;i<examples_vector->length-1;i++) // compute the objective function at every possible threshold (in between examples)
	{
		int32_t example_id=ordered[i];
		int32_t next_example_id=ordered[i+1];
		example_t* example=examples[example_id];
		//fprintf(stderr,"%zd %zd %f\n",i,vector_get_int32_t(ordered,i),vector_get_float(boostemplate->values,example_id));
		if(isnan(values[example_id]) || isnan(values[next_example_id]))break; // skip unknown values
		//example_t* next_example=(example_t*)vector_get(examples,(size_t)next_example_id);
		for(l=0;l<num_classes;l++) // update the objective function by putting the current example the other side of the threshold
		{
			weight[1][b(example,l)][l]+=example->weight[l];
			weight[2][b(example,l)][l]-=example->weight[l];
		}
		if(values[example_id]==values[next_example_id])continue; // same value
		double objective=0;
		for(l=0;l<num_classes;l++) // compute objective Z()
		{
            double w0 = weight[0][1][l]*weight[0][0][l]; if(w0 > 0) objective += SQRT(w0);
            double w1 = weight[1][1][l]*weight[1][0][l]; if(w1 > 0) objective += SQRT(w1);
            double w2 = weight[2][1][l]*weight[2][0][l]; if(w2 > 0) objective += SQRT(w2);
		}
		objective*=2;
		//fprintf(stderr,"DEBUG: column=%d threshold=%f obj=%f\n",column,(vector_get_float(next_example->features,column)+vector_get_float(example->features,column))/2,objective);
		if(objective-min_objective<-1e-11) // get argmin
		{
			classifier->objective=objective;
			classifier->threshold=((double)values[next_example_id]+(double)values[example_id])/2.0; // threshold between current and next example

			// fix continuous features with null variance -> this behavior is not compatible with boostexter
			/*if(isnan(values[next_example_id])) classifier->threshold = values[example_id] + 1.0;
			if(isnan(values[example_id])) classifier->threshold = values[next_example_id];*/

			if(isnan(classifier->threshold))
				die("threshold is nan, column=%d \"%s\", objective=%f, i=%zd, example_id=%d (%f) next_example_id=%d (%f)",column, boostemplate->name->data, objective,i, example_id, (double)values[example_id], next_example_id, (double)values[next_example_id]); // should not happend
			//fprintf(stderr," %d:%d:%f",column,i,classifier->threshold);
			min_objective=objective;
			for(l=0;l<num_classes;l++) // update class weight
			{
				classifier->c0[l]=0.5*LOG((weight[0][1][l]+epsilon)/(weight[0][0][l]+epsilon));
				classifier->c1[l]=0.5*LOG((weight[1][1][l]+epsilon)/(weight[1][0][l]+epsilon));
				classifier->c2[l]=0.5*LOG((weight[2][1][l]+epsilon)/(weight[2][0][l]+epsilon));
			}
		}
	}
	//fprintf(stderr,"DEBUG: column=%d threshold=%f obj=%f %s\n",column,classifier->threshold,classifier->objective,boostemplate->name->data);
	if(isnan(classifier->threshold)) // not found a better classifier
	{
		FREE(classifier->c0);
		FREE(classifier->c1);
		FREE(classifier->c2);
		FREE(classifier);
		return NULL;
	}
	return classifier;
}

double compute_max_error(vector_t* examples, int num_classes)
{
	double error=0;
	int i,l;
	for(i=0; i<examples->length; i++)
	{
		test_example_t* example = (test_example_t*)vector_get(examples, i);

		int max_class = 0;
		for(l=0;l<num_classes;l++) // selected class = class with highest score
		{
			if (example->score[l]>example->score[max_class]) max_class = l;
		}
		if (!b(example, max_class)) error++;
	}
	return error/(examples->length);
}

double compute_test_error(vector_t* classifiers, vector_t* examples, int classifier_id, int num_classes)
{
	int i;
	int l;
	double error=0;
	weakclassifier_t* classifier = (weakclassifier_t*)vector_get(classifiers, classifier_id);
	for(i=0; i<examples->length; i++)
	{
		test_example_t* example = (test_example_t*)vector_get(examples, i);
		if(classifier->type == CLASSIFIER_TYPE_THRESHOLD)
		{
			float value=example->continuous_features[classifier->column];
			if(isnan(value))
			{
				//fprintf(stderr, "%d %f==NAN\n", i+1, value, classifier->threshold);
				for(l=0;l<num_classes;l++)
				{
					example->score[l]+=classifier->alpha*classifier->c0[l];
				}
			}
			else if(value<classifier->threshold)
			{
				//fprintf(stderr, "%d %f<%f\n", i+1, value, classifier->threshold);
				for(l=0;l<num_classes;l++)
					example->score[l]+=classifier->alpha*classifier->c1[l];
			}
			else
			{
				//fprintf(stderr, "%d %f>=%f\n", i+1, value, classifier->threshold);
				for(l=0;l<num_classes;l++)
					example->score[l]+=classifier->alpha*classifier->c2[l];
			}
		}
		else if(classifier->type == CLASSIFIER_TYPE_TEXT)
		{
			int j;
			int has_token=0;
			//tokeninfo_t* tokeninfo=(tokeninfo_t*) vector_get(classifier->boostemplate->tokens,classifier->token);
			if(example->discrete_features[classifier->column] != NULL)
				for(j=0; j<example->discrete_features[classifier->column]->length; j++)
				{
					if(vector_get_int32_t(example->discrete_features[classifier->column], j)==classifier->token)
					{
						has_token=1;
						break;
					}
				}
			if(has_token)
			{
				//fprintf(stderr, "%d has token %s\n", i, tokeninfo->key);
				for(l=0;l<num_classes;l++)
					example->score[l]+=classifier->alpha*classifier->c2[l];
			}
			else // unknown or absent (c1 = c0)
			{
				//fprintf(stderr, "%d not has token %s\n", i, tokeninfo->key);
				for(l=0;l<num_classes;l++)
					example->score[l]+=classifier->alpha*classifier->c1[l];
			}
		}
		int erroneous_example = 0;
		//if(i<10)fprintf(stderr,"%d %f %f\n", i, example->score[0], example->score[1]);
		for(l=0;l<num_classes;l++) // selected class = class with highest score
		{
			if(example->score[l]>0.0 && !b(example,l)) erroneous_example = 1;
			else if(example->score[l]<=0.0 && b(example,l)) erroneous_example = 1;
		}
		if(erroneous_example == 1) error++;
	}
	return error/(examples->length);
}

int example_score_comparator(const void* a, const void* b)
{
	test_example_t* aa = *((test_example_t**)a);
	test_example_t* bb = *((test_example_t**)b);
	if(aa->score[example_score_comparator_class] > bb->score[example_score_comparator_class])
		return 1;
	else if(aa->score[example_score_comparator_class] < bb->score[example_score_comparator_class])
		return -1;
	return 0;
}

double compute_max_fmeasure(vector_t* working_examples, int class_of_interest, double* threshold, double *recall_output, double *precision_output)
{
	double maximum_fmeasure = -1;
	example_score_comparator_class = class_of_interest;
	vector_t* examples = vector_copy(working_examples);
	vector_sort(examples, example_score_comparator);
	int i;
	double true_below = 0;
	double total_true = 0;
	for(i=0; i<examples->length; i++)
	{
		test_example_t* example = (test_example_t*)vector_get(examples, i);
		if(b(example, class_of_interest)) total_true++;
	}
	double previous_value = NAN;
	for(i=0; i<examples->length; i++)
	{
		test_example_t* example = (test_example_t*)vector_get(examples, i);
		if(example->score[class_of_interest] != previous_value) {
			double precision = (total_true - true_below) / (examples->length - i);
			double recall = (total_true - true_below) / total_true;
			double fmeasure = fmeasure_beta * precision + recall > 0 ? (1 + fmeasure_beta) * recall * precision / (fmeasure_beta * precision + recall) : 0;
			if(fmeasure > maximum_fmeasure) {
				maximum_fmeasure = fmeasure;
				if(threshold != NULL) *threshold = (previous_value + example->score[class_of_interest]) / 2.0;
				if(recall_output != NULL) *recall_output = recall;
				if(precision_output != NULL) *precision_output = precision;
			}
			//fprintf(stderr, "EX: %d %f p=%f r=%f f=%f\n", i, example->score[class_of_interest], precision, recall, fmeasure);
			previous_value = example->score[class_of_interest];
		}
		if(b(example, class_of_interest)) true_below++;
	}
	vector_free(examples);
	return maximum_fmeasure;
}

double compute_classification_error(vector_t* classifiers, vector_t* examples, int classifier_id, double** sum_of_weights, int num_classes, int use_known_continuous_stump, int use_abstaining_text_stump)
{
	int i=0;
	int l=0;
	double error=0;
	double normalization=0;
	weakclassifier_t* classifier=(weakclassifier_t*)vector_get(classifiers,classifier_id);
	if(classifier->type==CLASSIFIER_TYPE_THRESHOLD)
	{
		for(i=0;i<examples->length;i++)
		{
			example_t* example=(example_t*)vector_get(examples,i);
			float value=vector_get_float(classifier->boostemplate->values,i);
			if(isnan(value))
			{
				if(use_known_continuous_stump == 0)
					for(l=0;l<num_classes;l++)
					{
						example->score[l]+=classifier->alpha*classifier->c0[l];
						example->weight[l]=example->weight[l]*EXP(-classifier->alpha*y_l(example,l)*classifier->c0[l]);
					}
			}
			else if(value<classifier->threshold)
			{
				for(l=0;l<num_classes;l++)
				{
					example->score[l]+=classifier->alpha*classifier->c1[l];
					example->weight[l]=example->weight[l]*EXP(-classifier->alpha*y_l(example,l)*classifier->c1[l]);
				}
			}
			else
			{
				for(l=0;l<num_classes;l++)
				{
					example->score[l]+=classifier->alpha*classifier->c2[l];
					example->weight[l]=example->weight[l]*EXP(-classifier->alpha*y_l(example,l)*classifier->c2[l]);
				}
			}
		}
	}
	else if(classifier->type==CLASSIFIER_TYPE_TEXT)
	{
		int* seen_examples = (int*)MALLOC(sizeof(int)*examples->length);
		memset(seen_examples,0,examples->length*sizeof(int));
		tokeninfo_t* tokeninfo=(tokeninfo_t*)vector_get(classifier->boostemplate->tokens,classifier->token);
		if(tokeninfo != NULL || tokeninfo->examples != NULL)
		{
			for(i=0;i<tokeninfo->examples->length;i++)
			{
				int32_t example_id=vector_get_int32_t(tokeninfo->examples,i);
				seen_examples[example_id]=1;
			}
		}
		for(i=0;i<examples->length;i++)
		{
			example_t* example=(example_t*)vector_get(examples,i);
			if(seen_examples[i]==1)
			{
				for(l=0;l<num_classes;l++)
				{
					example->score[l]+=classifier->alpha*classifier->c2[l];
					example->weight[l]=example->weight[l]*EXP(-classifier->alpha*y_l(example,l)*classifier->c2[l]);
				}
			}
			else // unknown or absent (c1 = c0)
			{
				if(use_abstaining_text_stump == 0)
					for(l=0;l<num_classes;l++)
					{
						example->score[l]+=classifier->alpha*classifier->c1[l];
						example->weight[l]=example->weight[l]*EXP(-classifier->alpha*y_l(example,l)*classifier->c1[l]);
					}
			}
		}
		FREE(seen_examples);
	}
	for(i=0;i<examples->length;i++)
	{
		example_t* example=(example_t*)vector_get(examples,i);
		int erroneous_example = 0;
		/*if(output_scores) {
			for(l=0;l<num_classes;l++) {
				fprintf(stderr,"%d ", example->classes[l]);
			}
		}
		if(output_scores) fprintf(stderr," scores:");*/
		for(l=0;l<num_classes;l++) // selected class = class with highest score
		{
			normalization+=example->weight[l]; // update Z() normalization (not the same Z as in optimization)
			if(example->score[l]>0.0 && !b(example,l)) erroneous_example = 1;
			else if(example->score[l]<=0.0 && b(example,l)) erroneous_example = 1;
			//if(output_scores) fprintf(stderr, " %f", example->score[l]/classifiers->length);
		}
		//if(output_scores) fprintf(stderr, " %s\n", erroneous_example ? "ERROR" : "OK" );
		if(erroneous_example == 1) error++;
	}
	if(sum_of_weights!=NULL)
	{
		//double min_weight=examples->length*num_classes;
		//double max_weight=0;
		//normalization/=num_classes*examples->length;
		for(l=0;l<num_classes;l++) // update the sum of weights by class
		{
			sum_of_weights[0][l]=0.0;
			sum_of_weights[1][l]=0.0;
		}
		for(i=0;i<examples->length;i++) // normalize the weights and do some stats for debugging
		{
			example_t* example=(example_t*)vector_get(examples,i);
			//fprintf(stderr,"%d",i);
			if(output_weights)fprintf(stderr,"iteration=%zd example=%d weights:\n", (size_t)classifier_id+1, i);
			for(l=0;l<num_classes;l++)
			{
				example->weight[l]/=normalization;
				if(output_weights)fprintf(stderr," %f",example->weight[l]);
				/*if(example->weight[l]<0)die("ERROR: negative weight: %d %d %f",i,l,example->weight[l]);
				  if(min_weight>example->weight[l]){min_weight=example->weight[l];}
				  if(max_weight<example->weight[l]){max_weight=example->weight[l];}*/
				//fprintf(stderr," %f",example->weight[l]);
				sum_of_weights[b(example,l)][l]+=example->weight[l];
			}
			if(output_weights)fprintf(stderr,"\n");
		}
		//fprintf(stderr,"norm=%.12f min=%.12f max=%.12f\n",normalization,min_weight,max_weight);
	}
	return error/(examples->length);
}

vector_t* load_examples_multilabel(const char* filename, vector_t* templates, vector_t* classes, double* class_priors,
	int feature_count_cutoff, int in_test)
{
	FILE* fp=stdin;
	if(strcmp(filename,"-")!=0)
	{
		fp=fopen(filename,"r");
		if(fp == NULL)
		{
			warn("can't load \"%s\"", filename);
			return NULL;
		}
	}
	string_t* line;
	vector_t* examples = vector_new(16);
	int line_num=0;
	int i,j;
	while((line=string_readline(fp))!=NULL)
	{
		if(line_num % 1000 == 0)fprintf(stderr, "\r%s: %d", filename, line_num);
		line_num++;
		string_chomp(line);
		if(string_match(line,"^(\\|| *$)","n")) // skip comments and blank lines
		{
			string_free(line);
			continue;
		}
		array_t* array_of_tokens=string_split(line, " *, *", NULL);
		if(array_of_tokens->length != templates->length+1)
			die("wrong number of columns (%zd), \"%s\", line %d in %s", array_of_tokens->length, line->data, line_num, filename);
		test_example_t* test_example=NULL;
		example_t* example=NULL;
		if(in_test)
		{
			test_example = (test_example_t*)MALLOC(sizeof(test_example_t));
			test_example->score = (double*)MALLOC(sizeof(double)*classes->length);
			for(i=0; i<classes->length; i++)test_example->score[i]=0.0;
			test_example->continuous_features = (float*)MALLOC(sizeof(float)*templates->length);
			memset(test_example->continuous_features,0,sizeof(float)*templates->length);
			test_example->discrete_features = (vector_t**)MALLOC(sizeof(vector_t*)*templates->length);
			memset(test_example->discrete_features,0,sizeof(vector_t*)*templates->length);
			test_example->classes=NULL;
			test_example->num_classes=0;
		}
		else
		{
			example = (example_t*)MALLOC(sizeof(example_t));
			memset(example, 0, sizeof(example_t));
		}
		for(i=0; i<templates->length; i++)
		{
			boostemplate_t* boostemplate = (boostemplate_t*)vector_get(templates, i);
			string_t* token = (string_t*)array_get(array_of_tokens, i);
			//fprintf(stderr,"%d %d >>> %s\n", line_num, i, token->data);
			if(boostemplate->type == FEATURE_TYPE_CONTINUOUS)
			{
				float value=NAN;// unknwon is represented by Not-A-Number (NAN)
				char* error_location=NULL;
				if(token->length==0 || strcmp(token->data,"?")) // if not unknown value
				{
					value = strtof(token->data, &error_location);
					if(error_location==NULL || *error_location!='\0')
						die("could not convert \"%s\" to a number, line %d, column %d (%s) in %s", token->data, line_num, i, boostemplate->name->data, filename);
				}
				if(!in_test) vector_push_float(boostemplate->values,value);
				else test_example->continuous_features[boostemplate->column]=value;
			}
			else if(boostemplate->type == FEATURE_TYPE_TEXT || boostemplate->type==FEATURE_TYPE_SET)
			{
				if(token->length==0 || strcmp(token->data,"?")) // if not unknown value
				{
					if(in_test)test_example->discrete_features[boostemplate->column]=vector_new_int32_t(16);
					hashtable_t* bag_of_words=hashtable_new();
					string_t* field_string=string_new(token->data);
					array_t* experts=text_expert(boostemplate, field_string);
                    if(boostemplate->type == FEATURE_TYPE_SET && experts->length != 1) {
                        die("column %d \"%s\" cannot handle space-separated value \"%s\", line %d in %s", i, boostemplate->name->data, field_string->data, line_num, filename);
                    }
					for(j=0; j<experts->length ; j++)
					{
						string_t* expert = (string_t*)array_get(experts, j);
						tokeninfo_t* tokeninfo = (tokeninfo_t*)hashtable_get(boostemplate->dictionary, expert->data, expert->length);
						if(tokeninfo == NULL && !in_test)
						{
							if(in_test)
                                tokeninfo = (tokeninfo_t*)vector_get(boostemplate->tokens,0); // default to the unknown token
							else if(boostemplate->type == FEATURE_TYPE_TEXT) // || boostemplate->type == FEATURE_TYPE_SET) // update the dictionary with the new token
							{
								tokeninfo = (tokeninfo_t*)MALLOC(sizeof(tokeninfo_t));
								tokeninfo->id = boostemplate->tokens->length;
								tokeninfo->key = strdup(expert->data);
								tokeninfo->count=0;
								tokeninfo->examples=vector_new_int32_t(16);
								hashtable_set(boostemplate->dictionary, expert->data, expert->length, tokeninfo);
								vector_push(boostemplate->tokens, tokeninfo);
							}
							else die("value \"%s\" was not described in the .names file, line %d, column %d (%s) in %s", expert->data, line_num, i, boostemplate->name->data, filename);
						}
						//vector_set_int32(example->features,i,tokeninfo->id);
						if(tokeninfo!=NULL && hashtable_get(bag_of_words, expert->data, expert->length)==NULL)
						{
							hashtable_set(bag_of_words, expert->data, expert->length, expert->data);
							if(!in_test)
							{
								tokeninfo->count++;
								vector_push_int32_t(tokeninfo->examples,(int32_t)examples->length); // inverted index
							}
							else
							{
								vector_push_int32_t(test_example->discrete_features[boostemplate->column], tokeninfo->id);
							}
						}
					}
					string_array_free(experts);
					string_free(field_string);
					hashtable_free(bag_of_words);
					if(in_test)vector_optimize(test_example->discrete_features[boostemplate->column]);
				}
				else
				{
					//vector_set_int32(example->features,i,0); // unknown token is 0 (aka NULL)
				}
				// FEATURE_TYPE_IGNORE
			}
		}
		string_t* last_token = (string_t*)array_get(array_of_tokens, templates->length);
		array_t* array_of_labels = string_split(last_token, "( *\\.$|  *)", NULL);
		//string_t* tmp = string_join_cstr("#", array_of_labels);
		//fprintf(stderr,"classes [%s]\n", tmp->data);
		if(array_of_labels == NULL || array_of_labels->length<1)
			die("wrong class definition \"%s\", line %d in %s", last_token->data, line_num, filename);
        if(array_of_labels->length > 1) 
            has_multiple_labels_per_example = 1;
		if(in_test)
		{
			//test_example->num_classes = array_of_labels->length;
			//test_example->classes = MALLOC(sizeof(int32_t) * test_example->num_classes);
			test_example->num_classes = classes->length;
			test_example->classes = (int32_t*)MALLOC(sizeof(int32_t) * classes->length);
			memset(test_example->classes, 0, sizeof(int32_t) * classes->length);
		}
		else
		{
			//example->num_classes = array_of_labels->length;
			//example->classes = MALLOC(sizeof(int32_t) * example->num_classes);
			example->classes = (int32_t*)MALLOC(sizeof(int32_t) * classes->length);
			example->num_classes = classes->length;
			memset(example->classes, 0, sizeof(int32_t) * classes->length);
		}
		for(i=0; i<array_of_labels->length; i++)
		{
			string_t* classtype = (string_t*)array_get(array_of_labels, i);
			for(j=0; j<classes->length; j++)
			{
				string_t* other = (string_t*)vector_get(classes, j);
				if(string_eq(classtype, other))
				{
					/*if(in_test) test_example->classes[i] = j;
					else example->classes[i] = j;*/
					if(in_test) test_example->classes[j] = 1;
					else example->classes[j] = 1;
					break;
				}
			}
			if(j == classes->length)
				die("undeclared class \"%s\", line %d in %s", classtype->data, line_num, filename);
		}
		string_array_free(array_of_labels);
		string_array_free(array_of_tokens);
		string_free(line);
		if(in_test) vector_push(examples, test_example);
		else vector_push(examples, example);
	}
	fprintf(stderr, "\r%s: %d\n", filename, line_num);
#ifdef USE_CUTOFF
	if(!in_test)
	{
		for(i=0;i<templates->length;i++)
		{
			boostemplate_t* boostemplate=(boostemplate_t*)vector_get(templates,i);
			if(boostemplate->tokens->length>1)
			{
				for(j=1; j<boostemplate->tokens->length; j++) // remove infrequent features
				{
					tokeninfo_t* tokeninfo=vector_get(boostemplate->tokens,j);
					tokeninfo->id=j;
					if(tokeninfo->count<boostemplate->feature_count_cutoff)
					{
						/*if(verbose)
							fprintf(stderr, "CUTOFF: \"%s\" %zd < %d\n", tokeninfo->key, tokeninfo->count, boostemplate->feature_count_cutoff);*/
						hashtable_remove(boostemplate->dictionary, tokeninfo->key, strlen(tokeninfo->key));
						if(tokeninfo->examples!=NULL)
							vector_free(tokeninfo->examples);
						FREE(tokeninfo->key);
						FREE(tokeninfo);
						memcpy(boostemplate->tokens->data+j*sizeof(void*),boostemplate->tokens->data+(boostemplate->tokens->length-1)*sizeof(void*),sizeof(void*));
						boostemplate->tokens->length--;
						j--;
					}
				}
			}
			vector_optimize(boostemplate->tokens);
			vector_optimize(boostemplate->values);
		}
	}
#endif
	// initialize weights and score
	if(!in_test)
	{
		if(class_priors != NULL)
		{
			int l;
			for(l=0; l<classes->length; l++) class_priors[l] = 0.0;
			for(i=0;i<examples->length;i++)
			{
				example_t* example=(example_t*)vector_get(examples,i);
				for(j=0; j<example->num_classes; j++)
				{
					if(example->classes[j]) class_priors[j] ++;
					//class_priors[example->classes[j]]++;
				}
			}
			for(l=0; l<classes->length; l++)
			{
				class_priors[l] /= examples->length;
				//string_t* classtype = (string_t*)vector_get(classes, l);
				/*if(verbose)
					fprintf(stderr,"CLASS PRIOR: %s %f\n", classtype->data, class_priors[l]);*/
			}
		}
		double normalization=0.0;
		for(i=0;i<examples->length;i++)
		{
			example_t* example=(example_t*)vector_get(examples,i);
			//fprintf(stdout,"%d %d %p\n",i,(int)vector_get(example->features,0), example->weight);
			example->weight=(double*)MALLOC(classes->length*sizeof(double));
			example->score=(double*)MALLOC(classes->length*sizeof(double));
			for(j=0;j<classes->length;j++)
			{
				if(enforce_anti_priors == 1)
				{
					if(b(example,j)==0)
						example->weight[j]=class_priors[j];
					else
						example->weight[j]=1.0;
				}
				else
					example->weight[j]=1.0;
				normalization+=example->weight[j];
				example->score[j]=0.0;
			}
		}
		for(i=0;i<examples->length;i++)
		{
			example_t* example=(example_t*)vector_get(examples,i);
			for(j=0;j<classes->length;j++)
				example->weight[j]/=normalization;
		}
	}
	vector_optimize(examples);
	fclose(fp);
	return examples;
}


} // end namespace
