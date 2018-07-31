#ifndef __MACROS_H__
#define __MACROS_H__

#ifdef DEBUG
#include <stdio.h>

#define DEBUG_PRINT(x) printf x
#else
#define DEBUG_PRINT(x)
#endif

#endif /* __MACROS_H__ */
