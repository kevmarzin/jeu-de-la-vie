
#ifndef COMPUTE_IS_DEF
#define COMPUTE_IS_DEF

#define TILE_SIZE 32
#define ALIVE_STATE 0xffff00ff
#define DEAD_STATE 0x0

typedef void (*void_func_t) (void);
typedef unsigned (*int_func_t) (unsigned);

extern void_func_t first_touch [];
extern int_func_t compute [];
extern char *version_name [];
extern unsigned opencl_used [];

extern unsigned version;

// déclaration de la matrice qui associe une tuile à un booleen,
// si vrai = tuile à calculer, sinon sauter la tuile
extern int **tile_calc;

extern void clean_compute ();

#endif
