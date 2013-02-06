#include "boost_internal.h"

namespace icsiboost{

vector_t* load_model(vector_t* templates, vector_t* classes, char* filename, int num_iterations_to_load)
{
	int i;
	mapped_t* input = mapped_load_readonly(filename);
	if(input == NULL) return NULL;

	vector_t* classifiers=vector_new(16);
	hashtable_t* templates_by_name=hashtable_new();
	for(i=0; i<templates->length; i++)
	{
		boostemplate_t* boostemplate=(boostemplate_t*)vector_get(templates,i);
		hashtable_set(templates_by_name, boostemplate->name->data, boostemplate->name->length, boostemplate);
		boostemplate->classifiers=vector_new(16);
	}

	string_t* line=NULL;
	int line_num=0;
	int num_classifiers=-1;
	weakclassifier_t* current=NULL;
	while((line=mapped_readline(input))!=NULL)
	{
		line_num++;
		if(string_match(line,"^ *$","n")) // skip blank lines
		{
			string_free(line);
			continue;
		}
		if(num_classifiers==-1)
		{
			num_classifiers=string_to_int32(line);
			if(num_iterations_to_load == -1) num_iterations_to_load = num_classifiers;
		}
		else if(current==NULL)
		{
			vector_t* groups=string_match(line," *([^ ]+) Text:(SGRAM|THRESHOLD):([^:]+):(.*)",NULL);
			if(groups)
			{
				current=(weakclassifier_t*)MALLOC(sizeof(weakclassifier_t));
				string_t* template_name = (string_t*)vector_get(groups,3);
				boostemplate_t* boostemplate = (boostemplate_t*)hashtable_get(templates_by_name, template_name->data, template_name->length);
				if(boostemplate==NULL)
                    die("invalid column boostemplate name \"%s\", line %d in %s", template_name->data, line_num, filename);
				string_t* alpha = (string_t*)vector_get(groups,1);
				current->alpha=string_to_double(alpha);
				if(isnan(current->alpha))
                    die("invalid alpha value \"%s\", line %d in %s", alpha->data, line_num, filename);
				current->boostemplate=boostemplate;
				vector_push(classifiers, current);
				vector_push(current->boostemplate->classifiers,current);
				current->column=boostemplate->column;
				current->threshold=NAN;
				current->objective=NAN;
				current->token=0;
				current->type=0;
				current->c0=NULL;
				current->c1=NULL;
				current->c2=NULL;
				string_t* word=(string_t*)vector_get(groups,4);
				if(string_eq_cstr((string_t*)vector_get(groups,2),"SGRAM"))
				{
					current->type=CLASSIFIER_TYPE_TEXT;
					if(boostemplate->type==FEATURE_TYPE_SET)
					{
						int32_t token_num=string_to_int32(word)+1;
						if(token_num>boostemplate->tokens->length)
                            die("invalid token number \"%s\", line %d in %s", word->data, line_num, filename);
						current->token=token_num;
					}
					else if(boostemplate->type==FEATURE_TYPE_TEXT)
					{
						tokeninfo_t* tokeninfo = (tokeninfo_t*)hashtable_get(boostemplate->dictionary, word->data, word->length);
						if(tokeninfo==NULL)
						{
							tokeninfo = (tokeninfo_t*)MALLOC(sizeof(tokeninfo_t));
							tokeninfo->id = boostemplate->tokens->length;
							tokeninfo->key = strdup(word->data);
							tokeninfo->count=0;
							tokeninfo->examples=vector_new(1);
							hashtable_set(boostemplate->dictionary, word->data, word->length, tokeninfo);
							vector_push(boostemplate->tokens, tokeninfo);
						}
						current->token = tokeninfo->id;
                        if(strchr(word->data, '#') != NULL) {
                            warn("ngram-like tokens found in model, did you set the correct -W and -N options?");
                        }
					}
                    //if(verbose) fprintf(stderr, "  SGRAM(%s) \"%s\" (id=%d)\n", current->boostemplate->name->data, word->data, current->token);
				}
				else if(string_eq_cstr((string_t*)vector_get(groups,2),"THRESHOLD"))
				{
					current->type=CLASSIFIER_TYPE_THRESHOLD;
				}
				else 
                    die("invalid classifier definition \"%s\", line %d in %s", line->data, line_num, filename);
				string_vector_free(groups);
			}
			else 
                die("invalid classifier definition \"%s\", line %d in %s", line->data, line_num, filename);
		}
		else if(current->c0==NULL)
		{
			current->c0 = (double*)MALLOC(sizeof(double)*classes->length);
			array_t* values=string_split(line," ",NULL);
			if(values==NULL || values->length!=classes->length)
                die("invalid weight distribution \"%s\", line %d in %s",line->data,line_num,filename);
			for(i=0; i<values->length; i++)
			{
				string_t* value = (string_t*)array_get(values,i);
				current->c0[i]=string_to_double(value);
				if(isnan(current->c0[i]))
                    die("invalid value in distribution \"%s\", line %d in %s", value->data, line_num, filename);
			}
			string_array_free(values);
			if(current->type==CLASSIFIER_TYPE_TEXT)
			{
				current->c1 = (double*)MALLOC(sizeof(double)*classes->length);
				memcpy(current->c1, current->c0, sizeof(double)*classes->length);
			}
		}
		else if(current->c1==NULL)
		{
			current->c1 = (double*)MALLOC(sizeof(double)*classes->length);
			array_t* values=string_split(line," ",NULL);
			if(values==NULL || values->length!=classes->length)
                die("invalid weight distribution \"%s\", line %d in %s",line->data,line_num,filename);
			for(i=0; i<values->length; i++)
			{
				string_t* value = (string_t*)array_get(values,i);
				current->c1[i]=string_to_double(value);
				if(isnan(current->c1[i]))
                    die("invalid value in distribution \"%s\", line %d in %s", value->data, line_num, filename);
			}
			string_array_free(values);
		}
		else if(current->c2==NULL)
		{
			current->c2 = (double*)MALLOC(sizeof(double)*classes->length);
			array_t* values=string_split(line," ",NULL);
			if(values==NULL || values->length!=classes->length)
                die("invalid weight distribution \"%s\", line %d in %s",line->data,line_num,filename);
			for(i=0; i<values->length; i++)
			{
				string_t* value = (string_t*)array_get(values,i);
				current->c2[i]=string_to_double(value);
				if(isnan(current->c2[i]))
                    die("invalid value in distribution \"%s\", line %d in %s", value->data, line_num, filename);
			}
			string_array_free(values);
			if(current->type==CLASSIFIER_TYPE_TEXT)
			{
				current=NULL;
				if(num_iterations_to_load > 0 && classifiers->length >= num_iterations_to_load){ string_free(line); break;} // clip classifiers
			}
		}
		else if(current->type==CLASSIFIER_TYPE_THRESHOLD)
		{
			current->threshold=string_to_double(line);
            //if(verbose) fprintf(stderr, "  THRESHOLD(%s) %g\n", current->boostemplate->name->data, current->threshold);
			if(isnan(current->threshold))
                die("invalid threshold \"%s\", line %d in %s", line->data, line_num, filename);
			current=NULL;
			if(num_iterations_to_load > 0 && classifiers->length >= num_iterations_to_load){ string_free(line); break; } // clip classifiers
		}
		else 
            die("invalid classifier definition \"%s\", line %d in %s", line->data, line_num, filename);
		string_free(line);
	}
	//if(verbose)fprintf(stdout,"LOADED_CLASSIFIERS %zd\n",classifiers->length);
	mapped_free(input);
	hashtable_free(templates_by_name);
	for(i=0; i<templates->length; i++)
	{
		boostemplate_t* boostemplate = (boostemplate_t*)vector_get(templates, i);
		vector_optimize(boostemplate->classifiers);
	}
	vector_optimize(classifiers);
	return classifiers;
}


void save_model(vector_t* classifiers, vector_t* classes, char* filename, int pack_model, int optimal_iterations)
{
	if(file_test(filename, "f"))
	{
		string_t* backup_name = string_new(filename);
		string_append_cstr(backup_name, ".previous");
		if(rename(filename, backup_name->data) == -1)
		{
			warn("could not backup previous model to \"%s\"", backup_name->data);
		}
		string_free(backup_name);
	}
	FILE* output=fopen(filename,"w");
	if(output==NULL){
		fprintf(stderr, "could not output model in \"%s\"",filename);
		return ;
	}
	int i;
	int num_classifiers=classifiers->length;
	if(pack_model && optimal_iterations==0)
	{
		hashtable_t* packed_classifiers=hashtable_new();
		for(i=0; i<classifiers->length; i++)
		{
			//if(optimal_iterations!=0 && i>optimal_iterations+1)break;
			weakclassifier_t* classifier = (weakclassifier_t*)vector_get(classifiers, i);
			string_t* identifier=string_sprintf("%d:%f:%d",classifier->column, classifier->threshold, classifier->token);
			weakclassifier_t* previous_classifier = (weakclassifier_t*)hashtable_get(packed_classifiers, identifier->data, identifier->length);
			if(previous_classifier!=NULL)
			{
				int l;
				for(l=0;l<classes->length;l++)
				{
					previous_classifier->c0[l]+=classifier->c0[l];
					previous_classifier->c1[l]+=classifier->c1[l];
					previous_classifier->c2[l]+=classifier->c2[l];
				}
				previous_classifier->alpha+=classifier->alpha;
				FREE(classifier->c0);
				FREE(classifier->c1);
				FREE(classifier->c2);
				FREE(classifier);
				vector_set(classifiers, i ,NULL);
				num_classifiers--;
			}
			else
			{
				hashtable_set(packed_classifiers, identifier->data, identifier->length, classifier);
			}
			string_free(identifier);
		}
		hashtable_free(packed_classifiers);
	}
	fprintf(output,"%d\n\n",optimal_iterations!=0 ? optimal_iterations : num_classifiers);
	for(i=0; i<classifiers->length; i++)
	{
		//if(optimal_iterations!=0 && i>optimal_iterations+1)break;
		weakclassifier_t* classifier=(weakclassifier_t*)vector_get(classifiers,i);
		if(classifier==NULL)continue;
		fprintf(output,"   %.12f Text:",classifier->alpha);
		if(classifier->type==CLASSIFIER_TYPE_THRESHOLD)
		{
			fprintf(output,"THRESHOLD:%s:\n\n",classifier->boostemplate->name->data);
			int l=0;
			for(l=0;l<classes->length;l++) fprintf(output,"%.10f ",classifier->c0[l]/classifier->alpha); fprintf(output,"\n\n");
			for(l=0;l<classes->length;l++) fprintf(output,"%.10f ",classifier->c1[l]/classifier->alpha); fprintf(output,"\n\n");
			for(l=0;l<classes->length;l++) fprintf(output,"%.10f ",classifier->c2[l]/classifier->alpha); fprintf(output,"\n\n");
			fprintf(output,"%.10f\n\n\n",classifier->threshold);
		}
		else if(classifier->type==CLASSIFIER_TYPE_TEXT && classifier->boostemplate->type==FEATURE_TYPE_TEXT)
		{
			tokeninfo_t* tokeninfo=(tokeninfo_t*) vector_get(classifier->boostemplate->tokens,classifier->token);
			fprintf(output,"SGRAM:%s:%s\n\n",classifier->boostemplate->name->data,tokeninfo->key);
			int l=0;
			for(l=0;l<classes->length;l++) fprintf(output,"%.10f ",classifier->c1[l]/classifier->alpha); fprintf(output,"\n\n");
			for(l=0;l<classes->length;l++) fprintf(output,"%.10f ",classifier->c2[l]/classifier->alpha); fprintf(output,"\n\n");
			fprintf(output,"\n");
		}
		else if(classifier->type==CLASSIFIER_TYPE_TEXT && classifier->boostemplate->type==FEATURE_TYPE_SET)
		{
			tokeninfo_t* tokeninfo=(tokeninfo_t*) vector_get(classifier->boostemplate->tokens,classifier->token);
			fprintf(output,"SGRAM:%s:%d\n\n",classifier->boostemplate->name->data,tokeninfo->id-1); // 0 is unknown (?), so skip it
			int l=0;
			for(l=0;l<classes->length;l++) fprintf(output,"%.10f ",classifier->c1[l]/classifier->alpha); fprintf(output,"\n\n");
			for(l=0;l<classes->length;l++) fprintf(output,"%.10f ",classifier->c2[l]/classifier->alpha); fprintf(output,"\n\n");
			fprintf(output,"\n");
		}
		else die("unknown classifier type \"%d\"",classifier->type);
	}
	fclose(output);
}


array_t* text_expert(boostemplate_t* boostemplate, string_t* text)
{
	array_t* words=string_split(text, " ", NULL);
	array_t* output = words;
	if(boostemplate->text_expert_length>1 && words->length>1)
	{
        output = array_new();
        int i,j;
        if(boostemplate->text_expert_type==TEXT_EXPERT_NGRAM)
        {
            for(i=0; i<words->length; i++)
            {
                array_t* gram=array_new();
                for(j=0; j<boostemplate->text_expert_length && i+j<words->length ;j++)
                {
                    array_push(gram, array_get(words, i+j));
                    array_push(output, string_join_cstr("#", gram));
                }
                array_free(gram);
            }
        }
        else if(boostemplate->text_expert_type==TEXT_EXPERT_FGRAM)
        {
            for(i=0; i+boostemplate->text_expert_length-1<words->length; i++)
            {
                array_t* gram=array_new();
                for(j=0; j<boostemplate->text_expert_length && i+j<words->length ;j++)
                {
                    array_push(gram, array_get(words, i+j));
                }
                array_push(output, string_join_cstr("#", gram));
                array_free(gram);
            }
        }
        else if(boostemplate->text_expert_type==TEXT_EXPERT_SGRAM) // different results than boostexter
        {
            for(i=0; i<words->length; i++)
            {
                array_t* gram=array_new();
                array_push(gram, array_get(words, i));
                array_push(output, string_copy((string_t*)array_get(words, i)));
                for(j=1; j<boostemplate->text_expert_length && i+j<words->length ;j++)
                {
                    array_push(gram, array_get(words, i+j));
                    array_push(output, string_join_cstr("#", gram));
                    array_pop(gram);
                }
                array_free(gram);
            }
        }
        else
        {
            die("unimplemented text expert");
        }
        string_array_free(words);
    }
    if(boostemplate->no_unk_ngrams)
    {
        array_t* no_unk=string_array_grep(output, "(^|#)unk(#|$)", "!");
        string_array_free(output);
        if(no_unk == NULL)
            return array_new();
        return no_unk;
    }
    if(boostemplate->drop_regex)
    {
        //if(verbose) fprintf(stderr, "DROP(%s) /%s/\n", boostemplate->name->data, boostemplate->drop_regex->data);
        array_t* filtered=string_array_grep(output, boostemplate->drop_regex->data, "!");
        if(filtered == NULL) filtered = array_new();
		/*
        if(verbose && filtered->length != output->length) {
            string_t* before = string_join_cstr(" ", output);
            string_t* after = string_join_cstr(" ", filtered);
            fprintf(stderr, "DROP [%s] => [%s]\n", before->data, after->data);
        }*/
        string_array_free(output);
        return filtered;
    }
    return output;
}

} // end namespace
