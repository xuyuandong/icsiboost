APP_PATH := $(PWD)

LIB_PATH := -L ./ -L /usr/lib64/ -L /usr/local/lib/ 

LIB := -lpcre -lpthread
#-lprotobuf -ltcmalloc -lprofiler

INCLUDE := -I ../include/ -I /usr/include/ 

#CXXFLAGS =-g -Wall -fPIC $(INCLUDE)
CXXFLAGS = -Wall -fPIC -O2 $(INCLUDE)

OBJ = array.o \
	boost_classifier.o \
	boost_internal.o \
	boost_training.o \
	common.o \
	debug.o \
	file.o \
	hashtable.o \
	mapped.o \
	string.o \
	threads.o \
	vector.o

LIBNAME=icsiboost.a

STLIB_MAKE_CMD=ar rcs $(LIBNAME) 

$(LIBNAME): $(OBJ)
	$(STLIB_MAKE_CMD) $(OBJ) 

clean:
	rm -rf $(LIBNAME) *.o 
