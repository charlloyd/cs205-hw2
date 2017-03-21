#include "argproc.h"

#ifdef __cplusplus
extern "C" {
#endif

// free an OptionTable object made by make_opttable()
void free_opttable(struct OptionTable *opttable)
{
    if (opttable) {
        if (opttable->table) {
            int i, size;
            struct Option *table = opttable->table;
            for (i = 0, size = opttable->size; i < size; i++) {
                if (table[i].name)
                    free(table[i].name);
                if (table[i].val)
                    free(table[i].val);
            }
            free(table);
        }
        free(opttable);
    }
}

// make an OptionTable object
struct OptionTable *make_opttable(int size, const char *names[], int flags[], int map[])
{
    struct OptionTable *opttable = (struct OptionTable *)malloc(sizeof(struct OptionTable));
    if (!opttable)
        return NULL;
    opttable->size = size;
    opttable->table = (struct Option *)malloc(sizeof(struct Option) * size);
    if (!opttable->table) {
        free(opttable);
        return NULL;
    }
    int i, j;
    for (i = 0; i < size; i++) {
    opttable->table[i].name = strdup(names[i]);
    if (!opttable->table[i].name) {
        for (j = i - 1; j >= 0; j--)
            free(opttable->table[j].name);
            free(opttable->table);
            return NULL;
        }
        opttable->table[i].val = NULL;
        opttable->table[i].flag = flags[i];
        opttable->table[i].map = map[i];
    }
    return opttable;
}

// find the element in table which matches argv[i]
int matchopt(int i, char *argv[], int size, struct Option *table)
{
    char *arg = argv[i];
    if (strncmp(arg, "--", 2) == 0) {
        arg += 2;
    } else if (strncmp(arg, "-", 1) == 0) {
        arg += 1;
    } else {
        // not a option
        return NOT_OPTION;
    }
    if (*arg == '\0')
        return ARG_SEP;
    int j;
    for (j = 0; j < size; j++) {
        char *name = table[j].name;
        if (strncmp(arg, name, strlen(name)) == 0)
            return j;
    }
    // unknown option
    return UNKNOWN_OPTION;
}

// argtake() return the number of arguments taken starting from i
int argtake(int i, int argc, char *argv[], struct Option *table_el, struct Option *table_el_map)
{
    if (table_el->flag == 0) {
        if (table_el_map->val == NULL)
            table_el_map->val = strdup(table_el_map->name);
    } else if (table_el->flag == 1) {
        char *arg = strchr(argv[i], '=');
        if (arg && arg[1] != '\0') {
            if (table_el_map->val)
                free(table_el_map->val);
            table_el_map->val = strdup(arg + 1);
        }
    } else if (table_el->flag == 2) {
        if (i + 1 < argc) {
            if (table_el_map->val)
                free(table_el_map->val);
            table_el_map->val = strdup(argv[i + 1]);
            return 2;
        }
        return LACK_VALUE;
    }
    return 1;
}

// receive arguments and argument table
// return the number of processed arguments
int argproc(int argc, char *argv[], struct OptionTable *opttable)
{
    int i, j, d;
    for (i = 1; i < argc; i += d) {
        j = matchopt(i, argv, opttable->size, opttable->table);
        switch (j) {
            case NOT_OPTION: return i;
            case ARG_SEP: return i + 1;
            case UNKNOWN_OPTION:
            // process the error argument
            // print help message
            return j;
        }
        int jmap = opttable->table[j].map;
        d = argtake(i, argc, argv, &(opttable->table[j]), &(opttable->table[jmap]));
        switch (d) {
            case LACK_VALUE:
            // process the error argument
            // print help message
            return d;
        }
    }
    return i;
}

#ifdef __cplusplus
}
#endif
