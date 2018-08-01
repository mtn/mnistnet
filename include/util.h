#ifndef __UTIL_H__
#define __UTIL_H__


// A vector (in the math sense, not a strechy buffer)
typedef struct {
    int len;
    double* elem;
} Vector;

// Initialize a vector
// TODO check if this is used
Vector* new_vector(int veclen);
// Take in a buffer and copy its values into a vector
// The buffer doesn't have to be owned (i.e. it can be on the stack)
// TODO check if this is actually being used
Vector* into_vector(int veclen, double* buffer);
// Create a buffer for an existing vector
void init_vector(Vector* v, int veclen);

void free_vector(Vector* v);


// A initializizer for arrays of doubles
void doublearr_init(Vector* v, double (*init_fn)());


#endif /* __UTIL_H__ */
