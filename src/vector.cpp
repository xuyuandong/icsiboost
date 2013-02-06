/* Copyright (C) (2007) (Benoit Favre) <favre@icsi.berkeley.edu>

This program is free software; you can redistribute it and/or 
modify it under the terms of the GNU Lesser General Public License 
as published by the Free Software Foundation; either 
version 2 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA. */

#include "vector.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

namespace icsiboost{

#define VECTOR_RESIZE_FACTOR 3/2
#define VECTOR_RESIZE_LIMIT (256*1024)


vector_implement_functions_for_type(float, 0.0);
vector_implement_functions_for_type(int32_t, 0);


// implementations for vector of void*
vector_t* vector_new(size_t initialSize) { return _vector_new(initialSize,sizeof(void*)); }
void vector_push(vector_t* v, void* value) { _vector_push(v,&value); }
void* vector_pop(vector_t* v) { void** output=(void**) _vector_pop(v); return output==NULL?NULL:*output; }
void* vector_remove_element(vector_t* v, size_t index) { void** output=(void**) _vector_remove_element(v, index); return output==NULL?NULL:*output; }
void* vector_shift(vector_t* v) { void** output=(void**) _vector_shift(v); return output==NULL?NULL:*output; }
void vector_unshift(vector_t* v, void* value) { _vector_unshift(v, &value); }
size_t vector_search(vector_t* v, void* value) { return _vector_search(v ,&value); }
size_t vector_search_sorted(vector_t* v, void* value, int (*comparator)(const void*,const void*)) { return _vector_search_sorted(v, &value, comparator); }
void* vector_get(vector_t* v, size_t index) { return *(void**) _vector_get(v, index); }
void vector_set(vector_t* v, size_t index, void* value) { _vector_set(v, index, &value); }
void vector_insert_element(vector_t* v, int index, void* value) { _vector_insert_element(v, index, &value); }

size_t vector_memory_size(vector_t* input)
{
	size_t size=sizeof(vector_t);
	size+=input->size*input->element_size;
	return size;
}

void _vector_resize(vector_t* v,size_t newSize)
{
	//fprintf(stderr,"resize %p %d\n",v, newSize);

	if(newSize==v->size)return;
	if(newSize<sizeof(int))newSize=sizeof(int);

	v->data=REALLOC(v->data,v->element_size*newSize);
	/*void* newData=MALLOC(v->element_size*newSize);
	if(v->data!=NULL)
	{
		memcpy(newData,v->data,v->length*v->element_size);
		FREE(v->data);
	}
	v->data=newData;*/

	if(v->data==NULL)
		warn("_vector_resize(%zd), reallocating vector from %zd", newSize, v->size);
	v->size=newSize;
}

void vector_optimize(vector_t* v)
{
	_vector_resize(v,v->length);
}

void vector_free(vector_t* v)
{
	FREE(v->data);
	FREE(v);
}

void _vector_push(vector_t* v, void* data)
{
	if(v->length>=v->size)
	{
		int newSize=v->size*VECTOR_RESIZE_FACTOR+1;
#ifdef VECTOR_RESIZE_LIMIT
		if((newSize*v->element_size)/VECTOR_RESIZE_LIMIT>=(v->element_size*v->size)/VECTOR_RESIZE_LIMIT+1)
			newSize = (((v->element_size*v->size)/VECTOR_RESIZE_LIMIT+1)*VECTOR_RESIZE_LIMIT)/v->element_size;
#endif
		_vector_resize(v,newSize);
		//_vector_resize(v,v->size*VECTOR_RESIZE_FACTOR+1);
	}
	memcpy((char*)v->data + v->length * v->element_size, data, v->element_size);
	v->length++;
}

void* _vector_pop(vector_t* v)
{
	if(v->length==0)return NULL;
	void* value = (char*)v->data + v->element_size * (v->length-1);
	v->length--;
	if(v->size>2 && v->length<v->size/2)_vector_resize(v,v->length+v->length/2); // value should still be allocated
	return value;
}

void* _vector_remove_element(vector_t* v, size_t index)
{
#ifdef DEBUG
	if(index>=v->length) warn("vector_remove_element(%zd), out-of-bounds, index>=%zd", index, v->length);
#endif
	void* element = (char*)v->data+index * v->element_size;
	if(index!=v->length-1)
		memmove((char*)v->data + index * v->element_size, (char*)v->data + (index+1) * v->element_size, v->element_size * (v->length - index-1));
	v->length--;
	if(v->size>2 && v->length<v->size/2)_vector_resize(v,v->length+v->length/2);
	return element;
}

void* _vector_shift(vector_t* v)
{
	if(v->length==0)return NULL;
	return _vector_remove_element(v,0);
}

void _vector_unshift(vector_t* v, void* value)
{
	if(v->length>0)
	{
		_vector_push(v, (char*)v->data + v->element_size*(v->length-1));
		memmove((char*)v->data + v->element_size, v->data, v->element_size*(v->length-2));
		memcpy(v->data,value,v->element_size);
	}
	else
	{
		_vector_push(v, value);
	}
}

void vector_append(vector_t* v,vector_t* u)
{
#ifdef DEBUG
	if(v->element_size!=u->element_size)warn("vector_append(%p,%p), different element size (%zd!=%zd)",v,u,v->element_size,u->element_size);
#endif
	if(v->length+u->length>=v->size)_vector_resize(v,(v->length+u->length)*VECTOR_RESIZE_FACTOR+1);
	memmove((char*)v->data + v->element_size * v->length, u->data, v->element_size*(u->length));
	v->length+=u->length;
}

void vector_prepend(vector_t* v,vector_t* u)
{
#ifdef DEBUG
	if(v->element_size!=u->element_size)warn("vector_prepend(%p,%p), different element size (%zd!=%zd)",v,u,v->element_size,u->element_size);
#endif
	if(v->length+u->length>=v->size)_vector_resize(v,(v->length+u->length)*VECTOR_RESIZE_FACTOR+1);
	memmove((char*)v->data+v->element_size*(u->length), v->data, v->element_size*(v->length));
	memmove(v->data,u->data,v->element_size*(u->length));
	v->length+=u->length;
}

void _vector_insert_element(vector_t* v, int index, void* value)
{
	if(index<0) _vector_unshift(v, value);
	else if(index>=v->length) _vector_push(v, value);
	else
	{
		if(v->length>=v->size) _vector_resize(v,v->size*VECTOR_RESIZE_FACTOR+1);
		memmove((char*)v->data + v->element_size * (index+1), (char*)v->data + v->element_size * index, v->element_size*(v->length-index-1));
		v->length++;
		memcpy((char*)v->data + v->element_size*index, value, v->element_size);
	}
}

void vector_insert(vector_t* v, size_t index, vector_t* peer)
{
#ifdef DEBUG
	if(v->element_size!=peer->element_size)warn("vector_insert(%p,%zd,%p), different element size (%zd!=%zd)",v,index,peer,
			v->element_size,peer->element_size);
#endif
	if(index<0) vector_prepend(v, peer);
	else if(index>=v->length) vector_append(v, peer);
	else {
		if(v->length+peer->length>=v->size)
		{
			_vector_resize(v,v->size+peer->size*VECTOR_RESIZE_FACTOR+1);
		}
		memmove((char*)v->data+v->element_size*(index+peer->length), (char*)v->data+v->element_size*(index),v->element_size*(v->length-index-1));
		v->length+=peer->length;
		memcpy((char*)v->data+v->element_size*index, peer->data, peer->length*v->element_size);
	}
}

vector_t* vector_fusion(vector_t* v, vector_t* u)
{
	vector_append(v,u);
	vector_free(u);
	return v;
}

void vector_reverse(vector_t* v)
{
	size_t i=0;
	size_t j=v->length-1;
	char tmp[v->element_size];
	for(;i<v->length/2;i++)
	{
		memcpy(tmp, (char*)v->data+v->element_size*i, v->element_size);
		memcpy((char*)v->data+v->element_size*i, (char*)v->data+v->element_size*j, v->element_size);
		memcpy((char*)v->data+v->element_size*j, tmp, v->element_size);
		j--;
	}
}

void vector_remove(vector_t* v, size_t from, size_t to)
{
	if(from<0)from=0;
	if(to>v->length)to=v->length;
	if(to<=from)return;
	if(to!=v->length)
	{
		memmove((char*)v->data+v->element_size*from, (char*)v->data+v->element_size*to, v->element_size*(to-from));
	}
	v->length-=to-from;
	if(v->length<v->size/2)_vector_resize(v,v->length+v->length/2);
}

vector_t* _vector_new(size_t initialSize, size_t element_size)
{
	vector_t* v = (vector_t*)MALLOC(sizeof(vector_t));
	if(v==NULL)return NULL;
	v->data=NULL;//MALLOC(v->element_size*initialSize);
	v->size=0;
	v->element_size=element_size;
	v->length=0;
	_vector_resize(v,initialSize);
	if(v->data==NULL)
	{
		FREE(v);
		return NULL;
	}
	return v;
}

vector_t* vector_subpart(vector_t* v, size_t from, size_t to)
{
	if(from<0)from=0;
	if(to>v->length)to=v->length;
	vector_t* output=vector_new(to-from);
	output->element_size=v->element_size;
	memcpy(output->data, (char*)v->data+v->element_size*from, (to-from)*v->element_size);
	output->length=to-from;
	return output;
}

void* _vector_get(vector_t* v, size_t index)
{
#ifdef DEBUG
	if(index>=v->length) warn("vector_get(%zd), out-of-bounds, index>=%zd", index, v->length);
#endif
	return (char*)v->data+v->element_size*index;
}

void _vector_set(vector_t* v, size_t index, void* value)
{
#ifdef DEBUG
	if(index>=v->length) warn("vector_set(%zd, %p), out-of-bounds, index>=%zd", index, value, v->length);
#endif
	memcpy((char*)v->data+v->element_size*index, value, v->element_size);
}

vector_t* vector_copy(vector_t* input)
{
	vector_t* output=_vector_new(input->length,input->element_size);
	memcpy(output->data,input->data,input->length*input->element_size);
	output->length=input->length;
	return output;
}

vector_t* vector_copy_and_duplicate_pointers(vector_t* input, size_t value_size)
{
	vector_t* output=_vector_new(input->length,input->element_size);
	size_t i;
	for(i=0;i<input->length;i++)
	{
		void* value=MALLOC(value_size);
		memcpy((char*)output->data + output->element_size*i, &value, sizeof(value));
		memcpy(*(void**)((char*)output->data + output->element_size*i), *(void**)((char*)input->data+input->element_size*i), value_size);
	}
	output->length=input->length;
	return output;
}

void vector_remove_duplicates(vector_t* v)
{
	size_t i,j=0;
	for(i=0;i<v->length && j<v->length;i++)
	{
		memmove((char*)v->data+v->element_size*i, (char*)v->data+v->element_size*j, v->element_size);
		j++;
		while(j<v->length && memcmp((char*)v->data+v->element_size*i, (char*)v->data+v->element_size*j, v->element_size)==0)j++;
	}
	v->length=i;
}

size_t _vector_search(vector_t* v, void* value)
{
	size_t i;
	for(i=0;i<v->length;i++)
	{
		if(memcmp((char*)v->data+v->element_size*i,value,v->element_size)==0)return i;
	}
	return -1;
}

size_t _vector_search_sorted(vector_t* v, void* value, int (*comparator)(const void*,const void*))
{
	void* found = bsearch(value,v->data,v->length,v->element_size,comparator);
	if (found != NULL)
        return ((char*)found - (char*)v->data)/v->element_size;
	return -1;
}

void vector_sort(vector_t* v,int (*comparator)(const void*,const void*))
{
	qsort(v->data,v->length,v->element_size,comparator);
}

void vector_apply(vector_t* v, void (*callback)(void* data, void* metadata),void* metadata)
{
	size_t i;
	for(i=0; i<v->length; i++)
        callback((char*)v->data+v->element_size*i,metadata);
}

void vector_freedata(void* data,void* metadata)
{
	FREE(*(void**)data);
}


} // end namespace
