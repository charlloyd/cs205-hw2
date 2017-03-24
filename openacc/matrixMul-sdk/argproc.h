/*
 *     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ARGPROC_H
#define ARGPROC_H

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// definition of Option class
struct Option {
	char *name;
	char *val;
	int flag;
	int map;
	// flag == 0: --help -h
	// flag == 1: --user=adsf -user=asdf
	// flag == 2: --pass asdf -pass asdf
};

// definition of OptionTable class
struct OptionTable {
	int size;
	struct Option *table;
};

// error codes
enum ARG_ERROR {
	LACK_VALUE = -1,
	NOT_OPTION = -2,
	UNKNOWN_OPTION = -3,
	ARG_SEP = -4
};

extern void free_opttable(struct OptionTable *opttable);
extern struct OptionTable *make_opttable(int size, const char *names[], int flags[], int map[]);
extern int argproc(int argc, char *argv[], struct OptionTable *opttable);

#ifdef __cplusplus
}
#endif

#endif
